from pydantic import BaseModel
from typing import Optional
from enum import Enum


class SampleMethod(str, Enum):
    greedy = "greedy"
    top_k = "top_k"
    top_p = "top_p"
    beam = "beam"


class CheckMethod(str, Enum):
    absolute = "absolute"
    relative = "relative"


class LlmQuery(BaseModel):
    code: str
    model_id: str


class LlmUpdateQuery(LlmQuery):
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


class LlmCheckQuery(LlmQuery):
    docstring: str
    check_method: Optional[CheckMethod] = CheckMethod.absolute

    update_query: Optional[LlmUpdateQuery] = None


class LlmCheckResponse(BaseModel):
    docstring_probability: float
    generated_docstring_probability: Optional[float] = None


class LlmUpdateResponse(BaseModel):
    updated_docstring: str
