from typing import Dict, Any, AsyncGenerator, Iterable, List, Tuple
import json
import git
import concurrent.futures
from multiprocessing import cpu_count
import os
import datetime
import ast
from difflib import SequenceMatcher
from dataclasses import dataclass


@dataclass
class Function:
    code_similarity: float
    docstring_similarity: float
    commit: str
    date: str
    code: str
    docstring: str
    code_updated: bool
    docstring_updated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_similarity": self.code_similarity,
            "docstring_similarity": self.docstring_similarity,
            "commit": self.commit,
            "date": self.date,
            "code": self.code,
            "docstring": self.docstring,
            "code_updated": self.code_updated,
            "docstring_updated": self.docstring_updated
        }

@dataclass
class ExampleFile:
    repo_index: str
    file_path: str
    functions: Dict[str, List[Function]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_index": self.repo_index,
            "file_path": self.file_path,
            "functions": {
                k: [f.to_dict() for f in v] 
                for k, v in self.functions.items()
            }
        }


class RepoScraper:

    METADATA_PATH: str = "repositories.json"
    CLONE_PATH: str = "cloned_repos"

    _repositories: Dict[str, Any]

    def __init__(self):
        with open(self.METADATA_PATH, "r") as file:
            self._repositories = json.load(file)

    async def scrape(self) -> AsyncGenerator[Dict[str, Any], None]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            for index, repo in enumerate(self._repositories):
                yield executor.submit(self._scrape_examples, repo, index)

    def _scrape_examples(self, repo: Dict[str, Any], index: int):
        for repo_file in self._scrape_repo_files(repo, index):

    def _extract_examples(self, repo_file: Dict[str, Any], index: int) -> Dict[str, Any]:
        repo_index = index
        file_path = repo_file['file_path']

        functions: Dict[str, List[Function]] = {}

        for version in repo_file['versions']:
            commit = version['commit']
            date = version['date']

            full_code = (
                version['blob']
                    .replace('\r\n', '\n')
                    .replace('\r', '\n')
                    .replace('\t', self.INDENTATION)
            )
            full_code_lines = full_code.split('\n')

            if len(full_code_lines) < 3:
                continue
            while len(full_code_lines) >= 3:
                try:
                    tree = ast.parse("\n".join(full_code_lines))
                except (SyntaxError, IndentationError) as exception:
                    line_number = exception.lineno
                    if line_number > len(full_code_lines) // 2:
                        full_code_lines = full_code_lines[:line_number-1]
                    else:
                        full_code_lines = full_code_lines[line_number:]
                else:
                    break

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

                if function_name in functions:
                    last_function_code = functions[function_name][-1].code
                    last_docstring = functions[function_name][-1].docstring

                    code_updated = last_function_code != function_code
                    docstring_updated = last_docstring != docstring

                    if not code_updated and not docstring_updated:
                        continue

                    code_similarity = SequenceMatcher(a=last_function_code, b=function_code).ratio()
                    docstring_similarity = SequenceMatcher(a=last_docstring, b=docstring).ratio()
                else:
                    code_similarity = None
                    docstring_similarity = None
                    code_updated = None
                    docstring_updated = None
                    functions[function_name] = []

                function = Function(
                    code_similarity=code_similarity,
                    docstring_similarity=docstring_similarity,
                    commit=commit,
                    date=date,
                    code=function_code,
                    docstring=docstring,
                    code_updated=code_updated,
                    docstring_updated=docstring_updated
                )
                functions[function_name].append(function)

        if len(functions) == 0:
            return None
        
        return ExampleFile(
            repo_index=repo_index,
            file_path=file_path,
            functions=functions
        )

    def _scrape_repo_files(self, repo: Dict[str, Any], index: int) -> Iterable[Dict[str, Any]]:
        local_directory = os.path.join(self.CLONE_PATH, repo["name"])
        repo = git.Repo.clone_from(
            repo["html_url"],
            local_directory,
            bare=True
        )
        main_head = repo.heads[0]

        py_files = self._find_all_py_files(main_head)
        py_file_versions = {
            py_file: self._find_all_blob_versions(py_file)
            for py_file in py_files
        }
        for py_file, versions in py_file_versions.items():
            repo_file = {
                "full_name": repo["full_name"],
                "file_path": py_file.path,
                "versions": [
                    {
                        "commit": commit.hexsha,
                        "date": datetime.fromtimestamp(commit.committed_date).isoformat(),
                        "blob": blob.data_stream.read().decode("utf-8")
                    }
                    for blob, commit in versions
                ]
            }

            yield repo_file

    def _find_all_py_files(self, main_head: git.Head) -> Iterable[git.Blob]:
        """Find all Python files in _main_head"""
        return (
            blob for blob in main_head.commit.tree.traverse()
            if blob.path.endswith(".py")
        )
    
    def _find_all_blob_versions(self, blob: git.Blob) -> List[Tuple[git.Blob, git.Commit]]:
        """Find all versions of a blob, skip if it doesnt exist in a commit."""
        versions = []
        last_version = None
        for commit in self._repo.iter_commits(self._main_head):
            try:
                blob_version = commit.tree[blob.path]
                if last_version is not None and blob_version.data_stream.read().decode("utf-8") == last_version.data_stream.read().decode("utf-8"):
                    continue 
                last_version = blob_version
            except KeyError:
                continue
            else:
                versions.append((blob_version, commit))
        return versions
