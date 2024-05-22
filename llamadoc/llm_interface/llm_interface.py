from typing import List, Tuple, Type, Iteratable
import requests
import json

from model import SampleMethod, CheckMethod, LlmQuery, LlmUpdateQuery, LlmCheckQuery, LlmCheckResponse, LlmUpdateResponse


class LlmInterface:

    with open("llm_config.json") as file:
        llm_config = json.load(file)

    server_address: str = llm_config["address"]
    server_port: int = llm_config["port"]

    def _do_request(
        self, 
        queries: Iteratable[LlmQuery],
        endpoint: str
    ) -> List[float]:
        response = requests.post(
            url=f"{self.server_address}:{self.server_port}{endpoint}",
            json=[query.model_dump() for query in queries]
        )
        response.raise_for_status()
        return response

    def check(self):
        ...

    def update(self):
        ...
