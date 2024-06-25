from __future__ import annotations
from pydantic import BaseModel, TypeAdapter, model_validator, field_validator
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC

class DeserializationMixin:

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseModel:
        return TypeAdapter(cls).validate_json(data)

class SampleMethod(str, Enum):
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM = "beam"


class TestMethod(str, Enum):
    PREDICTION = "prediction"
    DISTANCE = "distance"
    UPDATE = "update"


class DistanceFunction(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class GenerationParameters(BaseModel):
    """
    Allowed ranges:

    sample_method == greedy:
        early_stopping: None
        num_beams: None
        temperature: >= 0
        top_k: None
        top_p: None
        num_beam_groups: None
    sample_method == top_k:
        early_stopping: None
        num_beams: None
        temperature: >= 0
        top_k: > 0
        top_p: None
        num_beam_groups: None
    sample_method == top_p:
        early_stopping: None
        num_beams: None
        temperature: >= 0
        top_k: None
        top_p: (0, 1]
        num_beam_groups: None
    sample_method == beam:
        early_stopping: bool
        num_beams: > 0
        temperature: >= 0
        top_k: None
        top_p: None
        num_beam_groups: > 0
    """

    # Length parameters
    max_length: Optional[int] = None  # The maximum length of the sequence to be generated.
    max_new_tokens: Optional[int] = None  # The maximum number of new tokens to be generated, excluding the prompt.
    min_length: Optional[int] = None  # The minimum length of the sequence to be generated.

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, value):
        if value is not None and value <= 0:
            raise ValueError("max_length must be greater than 0.")
        return value
    
    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(cls, value):
        if value is not None and value <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")
        return value
    
    @field_validator("min_length")
    @classmethod
    def validate_min_length(cls, value):
        if value is not None and value <= 0:
            raise ValueError("min_length must be greater than 0.")
        return value

    @model_validator(mode='after')
    def validate_length_parameters(self):
        if self.max_length is not None and self.max_new_tokens is not None:
            raise ValueError("Only one of max_length and max_new_tokens can be set.")
        return self
    
    # Sampling parameters
    sample_method: Optional[SampleMethod] = SampleMethod.GREEDY  # The method used to sample the next token.
    early_stopping: Optional[bool] = None  # Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
    num_beams: Optional[int] = None  # Number of beams for beam search. 1 means no beam search.
    temperature: Optional[float] = None  # The value used to module the next token probabilities.
    top_k: Optional[int] = None  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_p: Optional[float] = None  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    num_beam_groups: Optional[int] = None  # Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.

    @field_validator("num_beams")
    @classmethod
    def validate_num_beams(cls, value):
        if value is not None and value <= 0:
            raise ValueError("num_beams must be greater than 0.")
        return value
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value):
        if value is not None and value <= 0:
            raise ValueError("temperature must be greater than 0.")
        return value
    
    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value):
        if value is not None and value <= 0:
            raise ValueError("top_k must be greater than 0.")
        return value
    
    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value):
        if value is not None and (value <= 0 or value > 1):
            raise ValueError("top_p must be in the range (0, 1].")
        return value
    
    @field_validator("num_beam_groups")
    @classmethod
    def validate_num_beam_groups(cls, value):
        if value is not None and value <= 0:
            raise ValueError("num_beam_groups must be greater than 0.")
        return value
    
    @model_validator(mode='after')
    def validate_sampling_parameters(self):
        if self.sample_method != SampleMethod.BEAM:
            if self.early_stopping is not None:
                raise ValueError("early_stopping must be None if sample_method is not beam.")
            if self.num_beams is not None:
                raise ValueError("num_beams must be None if sample_method is not beam.")
            if self.num_beam_groups is not None:
                raise ValueError("num_beam_groups must be None if sample_method is not beam.")
        elif self.sample_method != SampleMethod.TOP_P:
            if self.top_p is not None:
                raise ValueError("top_p must be None if sample_method is not top_p.")
        elif self.sample_method != SampleMethod.TOP_K:
            if self.top_k is not None:
                raise ValueError("top_k must be None if sample_method is not top_k.")
        return self

    # Penalty parameters
    repetition_penalty: Optional[float] = None  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: Optional[float] = None  # Exponential penalty to the length. 1.0 means no penalty.
    diversity_penalty: Optional[float] = None  #  The value is subtracted from a beams score if it generates a token already generated by another beam. Note that this penalty is only applied if num_beam_groups > 1.
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = None  # This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: (start_index, decay_factor) where start_index indicates where penalty starts and decay_factor represents the factor of exponential decay

    @field_validator("repetition_penalty")
    @classmethod
    def validate_repetition_penalty(cls, value):
        if value is not None and value < 0:
            raise ValueError("repetition_penalty must be greater than or equal to 0.")
        return value
    
    @field_validator("length_penalty")
    @classmethod
    def validate_length_penalty(cls, value):
        if value is not None and value < 0:
            raise ValueError("length_penalty must be greater than or equal to 0.")
        return value
    
    @field_validator("diversity_penalty")
    @classmethod
    def validate_diversity_penalty(cls, value):
        if value is not None and value < 0:
            raise ValueError("diversity_penalty must be greater than or equal to 0.")
        return value
    
    @field_validator("exponential_decay_length_penalty")
    @classmethod
    def validate_exponential_decay_length_penalty(cls, value):
        if value is not None:
            start_index, decay_factor = value
            if start_index < 0:
                raise ValueError("start_index must be greater than or equal to 0.")
            if decay_factor <= 0:
                raise ValueError("decay_factor must be greater than 0.")
        return value
    
    @model_validator(mode='after')
    def validate_penalty_parameters(self):
        if self.diversity_penalty is not None and self.num_beam_groups is None:
            raise ValueError("diversity_penalty must be None if num_beam_groups is None.")
        return self
        
    def to_torch_dict(self) -> Dict[str, Any]:
        generation_parameters = {
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'min_length': self.min_length,
            'temperature': self.temperature,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty,
            'exponential_decay_length_penalty': self.exponential_decay_length_penalty
        }

        if self.sample_method == SampleMethod.GREEDY:
            pass
        elif self.sample_method == SampleMethod.TOP_K:
            generation_parameters['do_sample'] = True
            generation_parameters['top_k'] = self.top_k
        elif self.sample_method == SampleMethod.TOP_P:
            generation_parameters['do_sample'] = True
            generation_parameters['top_p'] = self.top_p
        elif self.sample_method == SampleMethod.BEAM:
            generation_parameters['do_sample'] = True
            generation_parameters['num_beams'] = self.num_beams
            generation_parameters['early_stopping'] = self.early_stopping
            generation_parameters['num_beam_groups'] = self.num_beam_groups
            generation_parameters['diversity_penalty'] = self.diversity_penalty

        return generation_parameters


class TestParameters(BaseModel, ABC):
    generation_parameters: GenerationParameters
    test_threshold: Optional[float] = 1.0

    @field_validator("test_threshold")
    @classmethod
    def validate_test_threshold(cls, value):
        if value < 0:
            raise ValueError("test_threshold must be greater than or equal to 0.")
        return value


class PredictionTestParameters(TestParameters):
    weight_decay: float
    frequency_importance: float

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay(cls, value):
        if value < 0 or value > 1:
            raise ValueError("weight_decay must be in the range [0, 1].")
        return value
    
    @field_validator("frequency_importance")
    @classmethod
    def validate_frequency_importance(cls, value):
        if value < 0 or value > 1:
            raise ValueError("frequency_importance must be in the range [0, 1].")
        return value


class DistanceTestParameters(TestParameters):
    mid: str
    distance_function: DistanceFunction
    normalize: bool
    sample_many: bool


class UpdateTestParameters(TestParameters):
    pass


TestParametersType = Union[PredictionTestParameters, DistanceTestParameters, UpdateTestParameters]


class TestQuery(BaseModel):
    mid: str
    codes: List[str]
    docstrings: List[str]
    test_method: TestMethod
    test_parameters: TestParametersType

    @model_validator(mode='after')
    def validate_test_query(self):
        if len(self.codes) != len(self.docstrings):
            raise ValueError("The number of codes and docstrings must be equal.")
        return self
        

class TestResult(BaseModel):
    out_of_date: Optional[bool] = None
    updated_docstring: str
    docstring_score: Optional[float] = None
    updated_docstring_score: Optional[float] = None


class TestResponse(BaseModel, DeserializationMixin):
    results: List[TestResult]
