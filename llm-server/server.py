from fastapi import FastAPI

from model import LlmUpdateQuery, LlmCheckQuery, LlmCheckResponse, LlmUpdateResponse
from llm import Llm

server = FastAPI()    

@server.get("/")
async def hello():
    return {"message": "Hello this is the LLaMaDoc server!"}


@server.get("/model-ids")
async def model_ids():
    return {"model_ids": Llm.model_ids}


@server.post("/check", response_model=LlmCheckResponse)
async def check(query: LlmCheckQuery):
    return Llm.check(query)


@server.post("/update", response_model=LlmUpdateResponse)
async def update(query: LlmUpdateQuery):
    print(type(query))
    return Llm.update(query)
