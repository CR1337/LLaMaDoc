"""
This script scrapes repository metadata from GitHub and stores it in repositories.json.
"""

import requests
import json
from tqdm import tqdm
from typing import Any, Dict, List

class RepositoryScraper:

    ITEMS_PER_PAGE: int = 100
    # GitHub API has a limit of 1000 items per search result
    MAX_RESULT_ITEMS: int = 1000

    with open("github-auth.token", 'r') as file:
        AUTH_TOKEN: str = file.read().replace('\n', '')
    HEADERS: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    del AUTH_TOKEN

    REPO_URL: str = "https://api.github.com/search/repositories"

    _language: str
    _star_threshold: int
    _repositories: List[Dict[str, Any]]

    _total_item_count: int = None
    _page_count: int = None
    
    _scraped: bool

    def __init__(
        self, 
        star_threshold: int,
        language: str = "python"
    ):
        self._language = language
        self._star_threshold = star_threshold
        
        self._repositories = []

        self._scraped = False

    def _scrape_repository_page(self, page: int):
        parameters = {
            "q": f"language:{self._language} stars:>={self._star_threshold}",
            "per_page": self.ITEMS_PER_PAGE,
            "sort": "stars",
            "page": page
        }
        response = requests.get(
            self.REPO_URL, headers=self.HEADERS, params=parameters
        )
        result = response.json()
        self._repositories.extend(result['items'])
        if page == 1:
            self._total_item_count = min(
                result['total_count'], self.MAX_RESULT_ITEMS
            )
            self._page_count = (
                self._total_item_count // self.ITEMS_PER_PAGE 
                + (
                    1 
                    if self._total_item_count % self.ITEMS_PER_PAGE > 0 
                    else 0
                )
            )
        
    def _scrape_repository_pages(self):
        self._scrape_repository_page(1)
        for page in tqdm(
            range(2, self._page_count + 1), 
            desc="Reading Repositories",
            unit="pages",
            total=self._page_count,
            initial=1
        ):
            self._scrape_repository_page(page)

    def _add_contributors(self, repository: Dict[str, Any]) -> Dict[str, Any]:
        contributors_url = repository['contributors_url']
        response = requests.get(contributors_url, headers=self.HEADERS)
        contributors = response.json()
        repository['contributors'] = contributors
        return repository

    def scrape(self):
        self._scrape_repository_pages()
        self._scraped = True
        for repository in tqdm(
            self._repositories, 
            desc="Reading Contributors",
            unit="repositories"
        ):
            repository = self._add_contributors(repository)

    def save(self, filename: str):
        if not self._scraped:
            raise RuntimeError("Repositories have not been scraped yet.")
        with open(filename, 'w') as file:
            json.dump(self._repositories, file, indent=2)

    @property
    def repository_count(self) -> int:
        if not self._scraped:
            raise RuntimeError("Repositories have not been scraped yet.")
        return len(self._repositories)

if __name__ == "__main__":
    scraper = RepositoryScraper(
        star_threshold=1000
    )
    scraper.scrape()
    scraper.save("repositories.json")
    print(f"Scraped {scraper.repository_count} repositories.")
