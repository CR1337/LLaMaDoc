from typing import Any, Dict, List, Tuple
import requests
import json

from model import (
    SampleMethod, 
    LlmQuery, 
    LlmResponse, 
    GenerationParameters,
    CheckParameters
)


class LlmInterface:

    _server_address: str
    _server_port: int
    _defaults: Dict[str, Any]
    _model_ids: List[str]

    @property
    def server_address(self) -> str:
        return self._server_address
    
    @property
    def server_port(self) -> str:
        return self._server_port
    
    @property
    def model_ids(self) -> List[str]:
        return self._model_ids

    def __init__(self):
        with open("llm_config.json") as file:
            llm_config = json.load(file)

        self._server_address = llm_config["server"]["address"]
        self._server_port = llm_config["server"]["port"]
        self._defaults = llm_config["defaults"]

        self._model_ids = self._get_model_ids()
        assert len(self._model_ids) > 0, "No models found"

    def _get_model_ids(self) -> List[str]:
        response = requests.get(
            url=f"{self._server_address}:{self._server_port}/model-ids"
        )
        response.raise_for_status()
        return response.json()

    def _do_request(
        self, 
        query: LlmQuery,
        endpoint: str
    ) -> Dict[str, Any]:
        response = requests.post(
            url=f"{self._server_address}:{self._server_port}{endpoint}",
            json=[query.model_dump() for query in query]
        )
        response.raise_for_status()
        return response.json()

    def update(
        self,
        codes: List[str],
        docstrings: List[str],
        *,
        model_id: str | None = None,
        max_length: int | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        length_penalty: float | None = None,
        sample_method: str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        num_beams: int | None = None,
        early_stopping: bool | None = None,
        weight_decay: float | None = None,
        frequency_importance: float | None = None,
        test_threshold: float | None = None,
    ) -> List[Tuple[bool, str]]:
        assert len(codes) == len(docstrings), "len(codes) != len(docstrings)"
    
        if len(codes) == 0:
            return []

        if sample_method is None:
            sample_method = SampleMethod(self._defaults["update"]["sample_method"])
        else:
            sample_method = SampleMethod(sample_method)

        model_id = model_id or self._model_ids[0]
        weight_decay = weight_decay or self._defaults["weight_decay"]
        frequency_importance = frequency_importance or self._defaults["frequency_importance"]
        test_threshold = test_threshold or self._defaults["test_threshold"]

        assert model_id in self._model_ids, f"model_id {model_id} not found"
        assert 0 <= weight_decay <= 1, "weight_decay must be in [0, 1]"
        assert 0 <= frequency_importance <= 1, "frequency_importance must be in [0, 1]"
        assert test_threshold >= 0, "test_threshold must be >= 0"

        check_parameters = CheckParameters(
            weight_decay=weight_decay,
            frequency_importance=frequency_importance,
            test_threshold=test_threshold
        )
        
        generation_parameters = GenerationParameters(
            max_length=max_length,
            temperature=temperature, 
            repetition_penalty=repetition_penalty, 
            length_penalty=length_penalty,
            sample_method=sample_method,
            top_k=top_k, 
            top_p=top_p, 
            num_beams=num_beams, 
            early_stopping=early_stopping
        )

        check_query = LlmQuery(
            llm_id=model_id,
            codes=codes,
            docstrings=docstrings,
            check_parameters=check_parameters,
            generation_parameters=generation_parameters
        )

        response = self._do_request(check_query, "/update")
        response = LlmResponse.from_dict(response)
        return [
            (out_of_date, updated_docstring) for out_of_date, updated_docstring
            in zip(response.out_of_date, response.updated_docstrings)
        ]
