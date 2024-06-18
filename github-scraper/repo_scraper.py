from typing import Dict, Any


class RepoScraper:

    METADATA_PATH: str = "repositories.json"

    _repositories: Dict[str, Any]

    def __init__(self):
        ...