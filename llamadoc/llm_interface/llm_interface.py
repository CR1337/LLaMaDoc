from typing import Any, Dict, List, Tuple
import requests
import json

from llm_interface.model import (
    TestQuery, TestMethod, DistanceFunction, 
    DistanceTestParameters, GenerationParameters
)


class LlmInterface:

    _server_address: str
    _server_port: int
    _generative_model_ids: List[str]
    _embedding_model_ids: List[str]

    @property
    def server_address(self) -> str:
        return self._server_address
    
    @property
    def server_port(self) -> str:
        return self._server_port
    
    @property
    def generative_model_ids(self) -> List[str]:
        return self._generative_model_ids
    
    @property
    def embedding_model_ids(self) -> List[str]:
        return self._embedding_model_ids

    def __init__(self):
        with open("server_config.json") as file:
            server_config = json.load(file)

        self._server_address = server_config["address"]
        self._server_port = server_config["port"]

        self._server_address = "http://localhost"
        self._server_port = 8000

        self._generative_model_ids = self._get_generative_model_ids()
        assert len(self._generative_model_ids) > 0, "No models found"
    
        self._embedding_model_ids = self._get_embedding_model_ids()
        assert len(self._embedding_model_ids) > 0, "No models found"

    def _get_model_ids(self, model_type: str) -> List[str]:
        response = requests.get(
            url=f"{self._server_address}:{self._server_port}/models/{model_type}"
        )
        response.raise_for_status()
        return response.json()

    def _get_generative_model_ids(self) -> List[str]:
        return self._get_model_ids("generative")
    
    def _get_embedding_model_ids(self) -> List[str]:
        return self._get_model_ids("embedding")

    def _get_loaded_model_ids(self, model_type: str) -> Dict[str, bool]:
        response = requests.get(
            url=f"{self._server_address}:{self._server_port}/models/{model_type}/loaded"
        )
        response.raise_for_status()
        return response.json()
    
    def get_loaded_generative_model_ids(self) -> Dict[str, bool]:
        return self._get_loaded_model_ids("generative")
    
    def get_loaded_embedding_model_ids(self) -> Dict[str, bool]:
        return self._get_loaded_model_ids("embedding")
    
    def unload_model(self, model_id: str) -> None:
        response = requests.delete(
            url=f"{self._server_address}:{self._server_port}/unload/{model_id}"
        )
        response.raise_for_status()
    
    def _do_request(
        self, 
        query: TestQuery,
        endpoint: str
    ) -> Dict[str, Any]:
        response = requests.post(
            url=f"{self._server_address}:{self._server_port}{endpoint}",
            json=query.model_dump()
        )
        response.raise_for_status()
        return response.json()

    def update(
        self,
        codes: List[str],
        docstrings: List[str]
    ) -> List[Tuple[bool, str]]:
        assert len(codes) == len(docstrings), "len(codes) != len(docstrings)"
    
        if len(codes) == 0:
            return []

        gen_mid = self._generative_model_ids[0]
        emb_mid = self._embedding_model_ids[0]

        test_query = TestQuery(
            mid=gen_mid,
            codes=codes,
            docstrings=docstrings,
            test_method=TestMethod.UPDATE,
            test_parameters=DistanceTestParameters(
                generation_parameters=GenerationParameters(
                    max_length=64,
                    sample_method="greedy"
                )
            )
        )

        response = self._do_request(test_query, "/test")

        return [
            (r["out_of_date"], r["updated_docstring"]) for r
            in response["results"]
        ]
    
    def check(
        self,
        codes: List[str],
        docstrings: List[str]
    ) -> List[Tuple[bool, str]]:
        assert len(codes) == len(docstrings), "len(codes) != len(docstrings)"
    
        if len(codes) == 0:
            return []

        gen_mid = self._generative_model_ids[0]
        emb_mid = self._embedding_model_ids[0]

        test_query = TestQuery(
            mid=gen_mid,
            codes=codes,
            docstrings=docstrings,
            test_method=TestMethod.DISTANCE,
            test_parameters=DistanceTestParameters(
                test_threshold=1.0,
                mid=emb_mid,
                distance_function=DistanceFunction.EUCLIDEAN,
                normalize=True,
                generation_parameters=GenerationParameters(
                    max_length=64,
                    sample_method="greedy"
                )
            )
        )

        response = self._do_request(test_query, "/test")

        return [
            (r["out_of_date"], r["updated_docstring"]) for r
            in response["results"]
        ]
        