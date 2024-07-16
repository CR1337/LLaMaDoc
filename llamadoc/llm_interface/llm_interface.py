from typing import Any, Dict, List, Tuple
import requests
import json
import os

from llm_interface.model import (
    TestQuery, TestMethod, DistanceFunction, 
    DistanceTestParameters, GenerationParameters
)

# Set this to True for using the fintuned model or to False for using the original model
finetuned = False

# Parameters for the best performing test for each model
DISTANCE = {True: DistanceFunction.EUCLIDEAN, False: DistanceFunction.EUCLIDEAN}
NORMALIZE = {True: False, False: True}
SAMPLE_MANY = {True: True, False: True}
TEST_THRESHOLD = {True: 1.39, False: 1.1}


class LlmInterface:
    """
    Interface for interacting with the LLM server
    """

    _server_address: str
    _server_port: int
    _generative_model_ids: List[str]
    _embedding_model_ids: List[str]

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "server_config.json")) as file:
            server_config = json.load(file)

        self._server_address = server_config["address"]
        self._server_port = server_config["port"]
    
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
        """
        Update the docstrings of the given codes

        Args:
            codes: List of code snippets
            docstrings: List of docstrings for the given code snippets

        Returns:
            List of tuples where the first element is a boolean indicating whether the docstring was updated
            and the second element is the updated docstring
        """
        assert len(codes) == len(docstrings), "len(codes) != len(docstrings)"
    
        if len(codes) == 0:
            return []

        if finetuned:
            gen_mid = "checkpoints/finetuned_0"
        else:
            gen_mid = "google/codegemma-2b"

            
        emb_mid = self._embedding_model_ids[0]

        test_query = TestQuery(
            mid=gen_mid,
            codes=codes,
            docstrings=docstrings,
            test_method=TestMethod.UPDATE,
            test_parameters=DistanceTestParameters(
                mid=emb_mid,
                distance_function=DISTANCE[finetuned],
                normalize=NORMALIZE[finetuned],
                sample_many=SAMPLE_MANY[finetuned],
                generation_parameters=GenerationParameters(
                    max_length=512,
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
        """
        Check if the docstrings of the given codes are up to date

        Args:
            codes: List of code snippets
            docstrings: List of docstrings for the given code snippets

        Returns:
            List of tuples where the first element is a boolean indicating whether the docstring is up to date
            and the second element is the suggested updated docstring
        """
        assert len(codes) == len(docstrings), "len(codes) != len(docstrings)"
    
        if len(codes) == 0:
            return []

        if finetuned:
            gen_mid = "checkpoints/finetuned_0"
        else:
            gen_mid = "google/codegemma-2b"

        emb_mid = self._embedding_model_ids[0]

        test_query = TestQuery(
            mid=gen_mid,
            codes=codes,
            docstrings=docstrings,
            test_method=TestMethod.DISTANCE,
            test_parameters=DistanceTestParameters(
                test_threshold=TEST_THRESHOLD[finetuned],
                mid=emb_mid,
                distance_function=DISTANCE[finetuned],
                normalize=NORMALIZE[finetuned],
                sample_many=SAMPLE_MANY[finetuned],
                generation_parameters=GenerationParameters(
                    max_length=512,
                    sample_method="greedy"
                )
            )
        )

        response = self._do_request(test_query, "/test")

        return [
            (r["out_of_date"], r["updated_docstring"]) for r
            in response["results"]
        ]
        