"""
This file contains all the FastAPI endpoints for the LLaMaDoc server.
"""

from fastapi import FastAPI, status, BackgroundTasks
from fastapi.responses import Response, PlainTextResponse

import torch
import gc

from out_of_date_test.gpu_memory import gpu_memory_summary

from out_of_date_test.model import TestQuery, TestResponse, TestMethod, PredictionTestParameters, DistanceTestParameters, NoneTestParameters
from out_of_date_test.model_provider import ModelProvider
from out_of_date_test.prediction_test import PredictionTest
from out_of_date_test.distance_test import DistanceTest
from out_of_date_test.none_test import NoneTest

from evaluation.compute_predictions import compute_predictions

server = FastAPI()    

@server.get("/")
async def hello():
    return {"message": "Hello this is the LLaMaDoc server!"}


@server.get("/models/generative")
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
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print(gpu_memory_summary())
    
    try:
        if query.test_method == TestMethod.PREDICTION:
            test = PredictionTest(query.mid)
            parameters = PredictionTestParameters(
                generation_parameters=query.test_parameters.generation_parameters,
                test_threshold=query.test_parameters.test_threshold,
                weight_decay=query.test_parameters.weight_decay,
                frequency_importance=query.test_parameters.frequency_importance
            )
        elif query.test_method == TestMethod.DISTANCE:
            test = DistanceTest(query.mid)
            parameters = DistanceTestParameters(
                generation_parameters=query.test_parameters.generation_parameters,
                test_threshold=query.test_parameters.test_threshold,
                mid=query.test_parameters.mid,
                distance_function=query.test_parameters.distance_function,
                normalize=query.test_parameters.normalize,
                sample_many=query.test_parameters.sample_many
            )
        elif query.test_method == TestMethod.UPDATE:
            test = NoneTest(query.mid)
            parameters = NoneTestParameters(
                generation_parameters=query.test_parameters.generation_parameters
            )

        results = test.test(query.codes, query.docstrings, parameters)
    except torch.cuda.OutOfMemoryError:
        mem_summary = gpu_memory_summary(long=True)
        print(mem_summary)
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        return PlainTextResponse(
            content=mem_summary,
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE
        )
    
    del test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(gpu_memory_summary())
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")

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
    

def background_compute_predictions(index: int):
    try:
        compute_predictions(index)
    except torch.cuda.OutOfMemoryError:
        mem_summary = gpu_memory_summary(long=True)
        print(mem_summary)
        import traceback
        traceback.print_exc()
    
    
@server.post("/compute-predictions/{index}")
async def compute_predictions(background_tasks: BackgroundTasks, index: str):
    index = int(index)
    background_tasks.add_task(background_compute_predictions, index)
    return Response(status_code=status.HTTP_200_OK)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8000)