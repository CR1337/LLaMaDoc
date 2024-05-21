import json
import git
import os
from typing import List, Dict, Any, Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
import shutil
from datetime import datetime

CLONE_PATH: str = "cloned_repos"


class RepositoryScraper:

    _index: int
    _local_directory: str
    _repo: git.Repo
    _repo_name: str
    _main_head: git.Head

    def __init__(self, repo: Dict[str, Any], index: int):
        self._index = index
        self._local_directory = os.path.join(CLONE_PATH, repo["name"])
        self._repo = git.Repo.clone_from(
            repo["html_url"], 
            self._local_directory, 
            bare=True
        )
        self._main_head = self._repo.heads[0]
        self._repo_name = repo["name"]
        if not os.path.exists("extracted-py-files"):
            os.mkdir("extracted-py-files")
        if not os.path.exists(f"extracted-py-files/{self._repo_name}"):
            os.mkdir(f"extracted-py-files/{self._repo_name}")

    def __del__(self):
        shutil.rmtree(self._local_directory)

    def scrape(self):
        py_files = self._find_all_py_files()
        py_file_versions = {
            py_file: self._find_all_blob_versions(py_file)
            for py_file in py_files
        }
        for py_file, versions in py_file_versions.items():
            output = {
                "repo_index": self._index,
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

            short_file_path = py_file.path.split("/")[-1]
            with open(f"extracted-py-files/{self._repo_name}/{short_file_path}.json", "w") as file:
                json.dump(output, file, indent=2)
            

    def _find_all_py_files(self) -> Iterable[git.Blob]:
        """Find all Python files in _main_head"""
        return (
            blob for blob in self._main_head.commit.tree.traverse()
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


def executor_workload(repository: Tuple[Dict[str, Any], int]):
    RepositoryScraper(*repository).scrape()


def main():
    with open("filtered_repositories.json", "r") as file:
        repositories = json.load(file)[:2]  # TODO: Remove the slicing

    # with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
    #     repositories_iter = tqdm(zip(repositories, range(len(repositories))), desc="Scraping Repositories")
    #     for _ in executor.map(executor_workload, repositories_iter):
    #         pass

    repositories_iter = tqdm(zip(repositories, range(len(repositories))), desc="Scraping Repositories")
    for repo in repositories_iter:
        RepositoryScraper(*repo).scrape()

if __name__ == "__main__":
    main()
