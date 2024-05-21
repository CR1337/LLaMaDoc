from fastapi import FastAPI
from typing import List

from model import LlmQuery, LlmCheckResponse, LlmUpdateResponse
from llm import check, update

server = FastAPI()    

@server.get("/")
async def hello():
    return {"message": "Hello this is the LLaMaDoc server!"}


@server.post("/check", response_model=List[LlmCheckResponse])
async def check(queries: List[LlmQuery]):
    return check(queries)


@server.post("/update", response_model=List[LlmUpdateResponse])
async def update(queries: List[LlmQuery]):
    return update(queries)
