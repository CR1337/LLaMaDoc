import json
import asyncio
import aiohttp
from typing import Dict, Any, AsyncGenerator

from async_util import async_range, get_json_request


class MetadataScraper:

    ITEMS_PER_PAGE: int = 100
    MAX_RESULT_ITEMS: int = 1000
    PAGE_AMOUNT: int = MAX_RESULT_ITEMS // ITEMS_PER_PAGE

    REPO_URL: str = "https://api.github.com/search/repositories"

    LANGUAGE: str = "python"
    STAR_THRESHOLD: int = 1000

    with open("auth.token", 'r') as file:
        AUTH_TOKEN: str = file.read().replace('\n', '')
    HEADERS: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    del AUTH_TOKEN

    METADATA_PATH: str = "repositories.json"

    async def _scrape_repository_pages(self, session: aiohttp.ClientSession) -> AsyncGenerator[Dict[str, Any], None]:
        parameters_template = {
            "q": f"language:{self.LANGUAGE} stars:>={self.STAR_THRESHOLD}",
            "per_page": self.ITEMS_PER_PAGE,
            "sort": "stars"
        }

        async for page in async_range(1, self.PAGE_AMOUNT + 1):
            parameters = parameters_template.copy() | {"page": page}
            response = await get_json_request(
                session, self.REPO_URL, self.HEADERS, parameters
            )
            yield response['items']

    async def _scrape(self):
        repositories = []
        async with aiohttp.ClientSession() as session:
            async for repositories in self._scrape_repository_pages(session):
                repositories.extend(repositories)
        return repositories
    
    def scrape_repositories(self):
        repositories = asyncio.run(self._scrape())
        with open(self.METADATA_PATH, "w") as file:
            json.dump(repositories, file, indent=2)

if __name__ == "__main__":
    MetadataScraper().scrape_repositories()
