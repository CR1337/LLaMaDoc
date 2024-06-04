from typing import Any, Dict, List, Tuple, Iteratable
import requests
import json

from model import (
    SampleMethod, 
    CheckMethod, 
    LlmQuery, 
    LlmUpdateQuery, 
    LlmCheckQuery, 
    LlmCheckResponse, 
    LlmUpdateResponse,
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

    def check(
        self,
        model_id: str,
        codes: List[str],
        docstrings: List[str],
        *,
        check_method: str | None = None,
        max_length: int | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        length_penalty: float | None = None,
        sample_method: str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        num_beams: int | None = None,
        early_stopping: bool | None = None,
        gamma: float | None = None,
        use_weight_decay: bool | None = None,
        use_frequency_weights: bool | None = None
    ) -> List[Tuple[float, float | None]]:
        assert len(codes) == len(docstrings), "len(codes) != len(docstrings)"
        if len(codes) == 0:
            return []
        
        if check_method is None:
            check_method = CheckMethod(self._defaults["check"]["check_method"])
        else:
            check_method = CheckMethod(check_method)

        if sample_method is None:
            sample_method = SampleMethod(self._defaults["update"]["sample_method"])
        else:
            sample_method = SampleMethod(sample_method)

        if gamma is None:
            gamma = self._defaults["check"]["gamma"]
        if use_weight_decay is None:
            use_weight_decay = self._defaults["check"]["use_weight_decay"]
        if use_frequency_weights is None:
            use_frequency_weights = self._defaults["check"]["use_frequency_weights"]

        check_parameters = CheckParameters(
            check_method=check_method,
            gamma=gamma,
            use_weight_decay=use_weight_decay,
            use_frequency_weights=use_frequency_weights
        )
        
        if check_method == CheckMethod.relative:
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
        else:
            generation_parameters = None

        check_query = LlmCheckQuery(
            codes=codes,
            check_parameters=check_parameters,
            generation_parameters=generation_parameters
        )

        response = self._do_request(check_query, "/check")
        check_response = LlmCheckResponse.from_dict(response)
        if check_method == CheckMethod.relative:
            return [
                (docstring_probability, generated_docstring_probability)
                for docstring_probability, generated_docstring_probability
                in zip(
                    check_response.docstring_probabilities,
                    check_response.generated_docstring_probabilities
                )
            ]
        else:
            return [
                docstring_probability for docstring_probability
                in check_response.docstring_probabilities
            ]

    def update(
        self,
        model_id: str,
        codes: List[str],
        *,
        max_length: int | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        length_penalty: float | None = None,
        sample_method: str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        num_beams: int | None = None,
        early_stopping: bool | None = None
    ) -> List[str]:
        if len(codes) == 0:
            return []
        
        if sample_method is None:
            sample_method = SampleMethod(self._defaults["update"]["sample_method"])
        else:
            sample_method = SampleMethod(sample_method)
        
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
        update_query = LlmUpdateQuery(
            codes=codes,
            model_id=model_id,
            generation_parameters=generation_parameters
        )

        response = self._do_request(update_query, "/update")
        return LlmUpdateResponse.from_dict(response).updated_docstrings
