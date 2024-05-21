from pydantic import BaseModel
from typing import Optional
import json


with open("llm_config.json") as file:
    llm_config = json.load(file)


class LlmQuery(BaseModel):
    code: str
    docstring: str

    max_new_tokens: Optional[int] = llm_config["defaults"]["max_new_tokens"]
    num_beams: Optional[int] = llm_config["defaults"]["num_beams"]
    do_sample: Optional[bool] = llm_config["defaults"]["do_sample"]
    temperature: Optional[float] = llm_config["defaults"]["temperature"]
    top_k: Optional[int] = llm_config["defaults"]["top_k"]
    top_p: Optional[float] = llm_config["defaults"]["top_p"]
    repetition_penalty: Optional[float] = llm_config["defaults"]["repetition_penalty"]
    length_penalty: Optional[float] = llm_config["defaults"]["length_penalty"]


class LlmCheckResponse(BaseModel):
    docstring_probability: float


class LlmUpdateResponse(BaseModel):
    updated_docstring: str