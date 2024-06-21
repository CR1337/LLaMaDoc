from typing import Dict, Any, List, Tuple
import json
import git
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import cpu_count
import os
import datetime
import ast
from difflib import SequenceMatcher
import shutil
import peewee
from tqdm import tqdm
import traceback
import sys 
from uuid import uuid4
import os


DB_PATH: str = "examples.db"


class BaseModel(peewee.Model):
    class Meta:
        database = peewee.PooledSqliteDatabase(DB_PATH)

    id = peewee.BinaryUUIDField(primary_key=True, default=uuid4)


class Repository(BaseModel):
    class Meta:
        table_name = "repositories"

    full_name = peewee.CharField()
    size = peewee.IntegerField()
    open_issues = peewee.IntegerField()
    watchers = peewee.IntegerField()
    contributors = peewee.IntegerField()
    forks = peewee.IntegerField()
    contributions = peewee.IntegerField()
    stars = peewee.IntegerField()


class File(BaseModel):
    class Meta:
        table_name = "files"

    repo = peewee.ForeignKeyField(Repository, backref="files")
    path = peewee.CharField()


class Function(BaseModel):
    class Meta:
        table_name = "functions"

    file_ = peewee.ForeignKeyField(File, backref="functions")
    name = peewee.CharField()


class Version(BaseModel):
    class Meta:
        table_name = "versions"

    function = peewee.ForeignKeyField(Function, backref="versions")
    last_version = peewee.ForeignKeyField("self", null=True)
    commit = peewee.CharField()
    date = peewee.DateTimeField()
    code = peewee.TextField()
    docstring = peewee.TextField()
    code_updated = peewee.BooleanField(null=True)
    docstring_updated = peewee.BooleanField(null=True)
    code_similarity = peewee.FloatField(null=True)
    docstring_similarity = peewee.FloatField(null=True)


class RepoScraper:

    METADATA_PATH: str = "repositories.json"
    CLONE_PATH: str = "cloned_repos"

    INDENTATION: str = "    "

    def __init__(self):
        shutil.rmtree(DB_PATH, ignore_errors=True)
        shutil.rmtree(self.CLONE_PATH, ignore_errors=True)

    def scrape(self):
        with open(self.METADATA_PATH, "r") as file:
            repository_data = json.load(file)
        with ProcessPoolExecutor(cpu_count() // 2)  as executor:
            futures = [
                executor.submit(self._scrape_repository, repo_data, index)
                for index, repo_data in enumerate(repository_data)
            ]
            for future in list(tqdm(as_completed(futures), total=len(repository_data), desc="Scraping repositories")):
                future.result()

    def _scrape_repository(self, repo_data: Dict[str, Any], index: int):
        contributors = repo_data.get("contributors", [])

        repo = Repository.create(
            full_name=repo_data["full_name"],
            size=repo_data["size"],
            open_issues=repo_data["open_issues"],
            watchers=repo_data["watchers"],
            contributors=len(contributors),
            forks=repo_data["forks"],
            contributions=sum(contributor["contributions"] for contributor in contributors),
            stars=repo_data["stargazers_count"],
        )
        
        local_directory = os.path.join(self.CLONE_PATH, repo.full_name.replace("/", "_"))

        try:
            repo_object = git.Repo.clone_from(
                repo_data["html_url"] + ".git", local_directory, bare=True
            )
            head = repo_object.heads[0]

            py_blobs: List[git.Blob] = self._find_all_py_blobs(head)
            files = [
                File(repo=repo, path=blob.path)
                for blob in py_blobs
            ]
            File.bulk_create(files)

            py_blob_versions = (
                self._find_all_blob_versions(blob, repo_object, head)
                for blob in py_blobs
            )

            [
                self._extract_examples(file_, blob_versions)
                for file_, blob_versions in zip(files, py_blob_versions)
            ]
        
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)

        finally:
            repo_object.close()
            shutil.rmtree(local_directory)
        

    def _find_all_py_blobs(self, head: git.Head) -> List[git.Blob]:
        return [
            blob for blob in head.commit.tree.traverse()
            if blob.path.endswith(".py")
        ]
    
    def _find_all_blob_versions(self, blob: git.Blob, repo_object: git.Repo, head: git.Head) -> List[Tuple[git.Blob, git.Commit]]:
        versions = []
        last_version = None
        for commit in repo_object.iter_commits(head):
            try:
                blob_version = commit.tree[blob.path]
                if last_version is not None and blob_version.data_stream.read() == last_version.data_stream.read():
                    continue 
                last_version = blob_version
            except (KeyError, IndexError, ValueError, AssertionError):
                continue
            else:
                versions.append((blob_version, commit))
        return versions

    def _extract_examples(self, file_: File, blob_versions: List[Tuple[git.Blob, git.Commit]]):
        functions_data: Dict[str, List[Function]] = {}

        for blob, commit in blob_versions:
            blob_bytes = blob.data_stream.read()
            if blob_bytes.count(b'\n') < 3:
                continue
            try:
                full_code = (
                    blob_bytes
                        .decode('utf-8')
                        .replace('\r\n', '\n')
                        .replace('\r', '\n')
                        .replace('\t', self.INDENTATION)
                )
            except (UnicodeDecodeError, ValueError):
                continue
            full_code_lines = full_code.split('\n')

            try:
                tree = ast.parse("\n".join(full_code_lines))
            except (SyntaxError, IndentationError):
                continue


            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if self._is_trivial_function(node):
                    continue
                docstring = ast.get_docstring(node)
                if not docstring:
                    continue

                function_name = node.name

                start_line = node.lineno
                end_line = node.end_lineno

                function_code_lines = full_code_lines[start_line-1:end_line]
                if len(function_code_lines) == 0:
                    continue
                while all(l.startswith(self.INDENTATION) or l.isspace() for l in function_code_lines):
                    function_code_lines = [l[len(self.INDENTATION):] for l in function_code_lines]

                function_code = "\n".join(function_code_lines)
                docstring_delimiter = '"""' if '"""' in function_code else "'''"
                try:
                    docstring_start = function_code.index(docstring_delimiter)
                except ValueError:
                    continue
                docstring_end = function_code[docstring_start+1:].index(docstring_delimiter)
                function_code = function_code[:docstring_start] + function_code[docstring_start+1+docstring_end+3:]

                if function_name in functions_data:
                    last_function_code = functions_data[function_name][-1]["code"]
                    last_docstring = functions_data[function_name][-1]["docstring"]

                    code_updated = last_function_code != function_code
                    docstring_updated = last_docstring != docstring

                    if not code_updated and not docstring_updated:
                        continue

                    code_similarity = SequenceMatcher(a=last_function_code, b=function_code).quick_ratio()
                    docstring_similarity = SequenceMatcher(a=last_docstring, b=docstring).quick_ratio()
                else:
                    code_similarity = None
                    docstring_similarity = None
                    code_updated = None
                    docstring_updated = None
                    functions_data[function_name] = []

                date = datetime.datetime.fromtimestamp(commit.committed_date).isoformat()

                functions_data[function_name].append({
                    "code_similarity": code_similarity,
                    "docstring_similarity": docstring_similarity,
                    "commit": commit,
                    "date": date,
                    "code": function_code,
                    "docstring": docstring,
                    "code_updated": code_updated,
                    "docstring_updated": docstring_updated
                })

        functions = []
        versions = []
        for function_name, function_data in functions_data.items():
            if len(function_data) == 0:
                continue
            function = Function(file_=file_, name=function_name)
            functions.append(function)
            last_version = None
            for data in function_data:
                last_version = Version(
                    function=function,
                    last_version=last_version,
                    commit=data["commit"],
                    date=data["date"],
                    code=data["code"],
                    docstring=data["docstring"],
                    code_updated=data["code_updated"],
                    docstring_updated=data["docstring_updated"],
                    code_similarity=data["code_similarity"],
                    docstring_similarity=data["docstring_similarity"]
                )
                versions.append(last_version)

        Function.bulk_create(functions)
        Version.bulk_create(versions)

    def _is_trivial_function(self, node):
        # Filter out any docstrings at the start of the body
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Str):
            body = body[1:]

        # Now check if the remaining body contains only `pass` or `...`
        if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Expr)):
            if isinstance(body[0], ast.Pass):
                return True
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and body[0].value.value is Ellipsis:
                return True
        return False


if __name__ == "__main__":
    db = peewee.SqliteDatabase(DB_PATH)
    db.connect()
    db.drop_tables([Repository, File, Function, Version], safe=True)
    db.create_tables([Repository, File, Function, Version])
    db.close()
    
    scraper = RepoScraper()
    scraper.scrape()
    