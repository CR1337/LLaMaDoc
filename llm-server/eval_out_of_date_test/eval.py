import gc
import json
import logging
import warnings
from itertools import islice
from random import Random
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from out_of_date_test.distance_test import DistanceTest
from out_of_date_test.model import (
    GenerationParameters,
    SampleMethod, NoneTestParameters, 
    TestParameters, TestResult
)
from out_of_date_test.model_provider import ModelProvider
from out_of_date_test.none_test import NoneTest


logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


SEED: int = 1337
RANDOM_NUMBER_GENERATOR: Random = Random(SEED)
SAMPLE_SIZE: int = 512
BATCH_SIZE: int = 32
GENERATION_PARAMETERS: GenerationParameters = GenerationParameters(
    max_new_tokens=256,
    sample_method=SampleMethod.TOP_P,
    top_p=0.85,
    temperature=0.5,
)
TEST_DATA_PATH: str = "eval_out_of_date_test/test_data.json"
PREDICTION_CACHE_PATH: str = "cache/updated_docstrings.json"
try:
    with open(TEST_DATA_PATH, "r") as f:
        TEST_DATA: List[Dict[str, Any]] = RANDOM_NUMBER_GENERATOR.sample(
            [
                {'c': item['c'], 'd': item['d'], 'l': item['l']}
                for item in json.load(f)
            ],
            k=SAMPLE_SIZE
        )
    BATCHED_TEST_DATA: List[List[Dict[str, Any]]] = list(batched(TEST_DATA, BATCH_SIZE))
except FileNotFoundError:
    TEST_DATA: List[Dict[str, Any]] = []
    BATCHED_TEST_DATA: List[List[Dict[str, Any]]] = []
    print("No test data found")

N_BATCHES: int = len(BATCHED_TEST_DATA)


def perform_test(
    mid: str, 
    precaching: bool, 
    test_parameters: TestParameters,
    updated_docstrings: List[str] | None = None
) -> List[TestResult]:
    if precaching:
        test = NoneTest(mid=mid)
    else:
        test = DistanceTest(mid=mid)
    test_results = []
    for batch in tqdm(
        BATCHED_TEST_DATA, 
        total=N_BATCHES
    ):
        print(len(batch))
        test_results.extend(
            test.test(
                [item['c'] for item in batch],
                [item['d'] for item in batch],
                test_parameters,
                updated_docstrings
            )
        )
    del test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_results


def compute_predictions() -> Dict[str, List[List[str]]]:
    test_parameters = NoneTestParameters(
        generation_parameters=GENERATION_PARAMETERS
    )
    mids = list(reversed(ModelProvider.generative_model_ids))
    results = {}
    for mid in tqdm(mids, total=len(mids), desc="Precaching"):
        print(f"Do precaching for {mid}")
        results[mid] = [
            [
                r.updated_docstring
                for r in perform_test(
                    mid=mid,
                    precaching=True,
                    test_parameters=test_parameters,
                    updated_docstrings=None
                )
            ]
            for _ in range(DistanceTest.N_SAMPLES)
        ]

    with open(PREDICTION_CACHE_PATH, "w") as f:
        json.dump(results, f)
    