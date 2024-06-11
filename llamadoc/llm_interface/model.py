from __future__ import annotations
from pydantic import BaseModel, TypeAdapter
from typing import Any, Dict, List, Optional
from enum import Enum

class DeserializationMixin:

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseModel:
        return TypeAdapter(cls).validate_json(data)

class SampleMethod(str, Enum):
    greedy = "greedy"
    top_k = "top_k"
    top_p = "top_p"
    beam = "beam"


class GenerationParameters(BaseModel):
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None

    sample_method: Optional[SampleMethod] = SampleMethod.greedy

    # top_k sampling
    top_k: Optional[int] = None

    # top_p sampling
    top_p: Optional[float] = None

    # beam search
    num_beams: Optional[int] = None
    early_stopping: Optional[bool] = None  

    def to_torch_dict(self):
        generation_parameters = {
            'max_length': self.max_length,
            'temperature': self.temperature,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty
        }

        if self.sample_method == SampleMethod.greedy:
            pass
        elif self.sample_method == SampleMethod.top_k:
            generation_parameters['top_k'] = self.top_k
        elif self.sample_method == SampleMethod.top_p:
            generation_parameters['top_p'] = self.top_p
        elif self.sample_method == SampleMethod.beam:
            generation_parameters['num_beams'] = self.num_beams
            generation_parameters['early_stopping'] = self.early_stopping

        return generation_parameters
    

class CheckParameters(BaseModel):
    weight_decay: float
    frequency_importance: float
    test_threshold: float


class LlmQuery(BaseModel):
    llm_id: str
    codes: List[str]
    docstring: str
    check_parameters: CheckParameters
    generation_parameters: GenerationParameters


class LlmResponse(BaseModel, DeserializationMixin):
    docstring_probabilities: List[float]
    generated_docstring_probabilities: List[float]
    out_of_date: List[bool]
    updated_docstrings: List[str]
