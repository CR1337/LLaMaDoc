from fastapi import FastAPI
from typing import List

from model import LlmUpdateQuery, LlmCheckQuery, LlmCheckResponse, LlmUpdateResponse
from llm import check, update, model_ids, get_special_tokens

server = FastAPI()    

@server.get("/")
async def hello():
    return {"message": "Hello this is the LLaMaDoc server!"}


@server.get("/model-ids")
async def model_sizes():
    return {"model_ids": model_ids}


@server.post("/check", response_model=List[LlmCheckResponse])
async def check(queries: List[LlmCheckQuery]):
    return check(queries)


@server.post("/update", response_model=List[LlmUpdateResponse])
async def update(queries: List[LlmUpdateQuery]):
    return update(queries)
