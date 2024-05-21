from typing import List, Tuple
import requests
import json

from model import LlmQuery, LlmCheckResponse, LlmUpdateResponse


with open("llm_config.json") as file:
    llm_config = json.load(file)


class LlmInterface:

    server_address: str = llm_config["server"]["address"]
    server_port: int = llm_config["server"]["port"]

    def _do_request(
        self, queries: List[Tuple[str, str]], endpoint: str
    ) -> List[float]:
        response = requests.post(
            url=f"{self.server_address}:{self.server_port}{endpoint}",
            json=[
                LlmQuery(code, docstring).model_dump()
                for code, docstring in queries
            ]
        )
        response.raise_for_status()
        return response

    def check(self, queries: List[Tuple[str, str]]) -> List[float]:
        response = self._do_request(queries, "/check")
        return [
            LlmCheckResponse(**r).docstring_probability
            for r in response.json()
        ]

    def update(self, queries: List[Tuple[str, str]]) -> str:
        response = self._do_request(queries, "/update")
        return [
            LlmUpdateResponse(**r).updated_docstring
            for r in response.json()
        ]
