from __future__ import annotations
from multiprocessing import Lock
from typing import List, Any, Dict
from abc import abstractmethod
import json
import os
import shutil
from uuid import uuid4


class BaseModel:
    N_FILES: int = 1
    FILE_LOCK: Any

    DATA_DIRECTORY: str = "data"
    FILENAME_PREFIX: str = "BASE_MODEL_"

    @classmethod
    @abstractmethod
    def initialize(cls):
        if os.path.exists(cls.DATA_DIRECTORY):
            shutil.rmtree(cls.DATA_DIRECTORY)
        os.makedirs(cls.DATA_DIRECTORY)
        cls.FILE_LOCK = Lock()

    @classmethod
    def create(cls, *args, **kwargs) -> BaseModel:
        instance = cls(*args, **kwargs)
        with cls.FILE_LOCK:
            filename = f"{cls.DATA_DIRECTORY}/{cls.FILENAME_PREFIX}{instance.id % cls.N_FILES}.json"
            with open(filename, "a") as f:
                f.write(json.dumps(instance.dict()) + "\n")
        return instance
    
    @classmethod
    def bulk_create(cls, instances: List[BaseModel]) -> List[BaseModel]:
        with cls.FILE_LOCK:
            filenames = (f"{cls.DATA_DIRECTORY}/{cls.FILENAME_PREFIX}{i}.json" for i in range(cls.N_FILES))
            fs = [open(filename, "a") for filename in filenames]
            [fs[instance.id % cls.N_FILES].write(json.dumps(instance.dict()) + "\n") for instance in instances]
            [f.close() for f in fs]
        return instances

    id: int

    def __init__(self):
        self.id = uuid4().int

    def dict(self) -> Dict[str, Any]:
        return {"id": self.id}


class Repository(BaseModel):
    N_FILES: int = 8
    FILENAME_PREFIX: str = "repositories_"

    full_name: str
    size: int
    open_issues: int
    watchers: int
    contributors: int
    forks: int
    contributions: int
    stars: int

    def __init__(
        self, 
        full_name: str, 
        size: int, 
        open_issues: int, 
        watchers: int,
        contributors: int, 
        forks: int, 
        contributions: int, 
        stars: int
    ):
        super().__init__()
        self.full_name = full_name
        self.size = size
        self.open_issues = open_issues
        self.watchers = watchers
        self.contributors = contributors
        self.forks = forks
        self.contributions = contributions
        self.stars = stars

    def dict(self) -> Dict[str, Any]:
        return {
            **super().dict(),
            "full_name": self.full_name,
            "size": self.size,
            "open_issues": self.open_issues,
            "watchers": self.watchers,
            "contributors": self.contributors,
            "forks": self.forks,
            "contributions": self.contributions,
            "stars": self.stars
        }


class File(BaseModel):
    N_FILES: int = 16
    FILENAME_PREFIX: str = "files_"

    path: str
    repo: int

    def __init__(self, path: str, repo: int):
        super().__init__()
        self.path = path
        self.repo = repo

    def dict(self) -> Dict[str, Any]:
        return {
            **super().dict(),
            "path": self.path,
            "repo": self.repo
        }


class Function(BaseModel):
    N_FILES: int = 32
    FILENAME_PREFIX: str = "functions_"

    name: str
    file: int

    def __init__(self, name: str, file: int):
        super().__init__()
        self.name = name
        self.file = file

    def dict(self) -> Dict[str, Any]:
        return {
            **super().dict(),
            "name": self.name,
            "file": self.file
        }


class Version(BaseModel):
    N_FILES: int = 64
    FILENAME_PREFIX: str = "versions_"

    commit: str
    date: str
    code: str
    docstring: str
    code_updated: bool | None
    docstring_updated: bool | None
    code_similarity: float | None
    docstring_similarity: float | None
    function: int
    last_version: int | None

    def __init__(
        self,
        commit: str,
        date: str,
        code: str,
        docstring: str,
        code_updated: bool | None,
        docstring_updated: bool | None,
        code_similarity: float | None,
        docstring_similarity: float | None,
        function: int,
        last_version: int | None
    ):
        super().__init__()
        self.commit = commit
        self.date = date
        self.code = code
        self.docstring = docstring
        self.code_updated = code_updated
        self.docstring_updated = docstring_updated
        self.code_similarity = code_similarity
        self.docstring_similarity = docstring_similarity
        self.function = function
        self.last_version = last_version

    def dict(self) -> Dict[str, Any]:
        return {
            **super().dict(),
            "commit": self.commit,
            "date": self.date,
            "code": self.code,
            "docstring": self.docstring,
            "code_updated": self.code_updated,
            "docstring_updated": self.docstring_updated,
            "code_similarity": self.code_similarity,
            "docstring_similarity": self.docstring_similarity,
            "function": self.function,
            "last_version": self.last_version
        }
