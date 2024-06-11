from fastapi import FastAPI, status
from fastapi.responses import Response

from out_of_date_test.model import TestQuery, TestResponse, TestMethod
from out_of_date_test.model_provider import ModelProvider
from out_of_date_test.prediction_test import PredictionTest
from out_of_date_test.distance_test import DistanceTest
from out_of_date_test.none_test import NoneTest

server = FastAPI()    

@server.get("/")
async def hello():
    return {"message": "Hello this is the LLaMaDoc server!"}


@server.get("/models/generation")
async def get_generative_models():
    return ModelProvider.generative_model_ids


@server.get("/models/embedding")
async def get_embedding_models():
    return ModelProvider.embedding_model_ids


@server.get("/models/generative/loaded")
async def get_loaded_models():
    return ModelProvider.get_loaded_generative_models()


@server.get("/models/embedding/loaded")
async def get_loaded_models():
    return ModelProvider.get_loaded_embedding_models()


@server.post("/test")
async def check(query: TestQuery) -> TestResponse:
    if query.mid not in ModelProvider.generative_model_ids:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    
    if query.test_method == TestMethod.PREDICTION:
        test = PredictionTest(query.mid)
    elif query.test_method == TestMethod.DISTANCE:
        test = DistanceTest(query.mid)
    elif query.test_method == TestMethod.UPDATE:
        test = NoneTest(query.mid)
    
    results = test.test(query.codes, query.docstrings, query.test_parameters)
    return TestResponse(results=results)


@server.delete("/unload/{mid:path}")
async def unload(mid: str):
    if mid in ModelProvider.generative_model_ids:
        ModelProvider.unload_generative_model(mid)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    elif mid in ModelProvider.embedding_model_ids:
        ModelProvider.unload_embedding_model(mid)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return Response(status_code=status.HTTP_404_NOT_FOUND)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8000)