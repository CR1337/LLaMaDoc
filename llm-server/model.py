from pydantic import BaseModel
from typing import Optional


class LlmQuery(BaseModel):
    code: str
    docstring: str

    max_new_tokens:int
    num_beams:int
    do_sample:bool
    temperature:float
    top_k:int
    top_p:float
    repetition_penalty:float
    length_penalty:float


class LlmCheckResponse(BaseModel):
    docstring_probability: float


class LlmUpdateResponse(BaseModel):
    updated_docstring: str
