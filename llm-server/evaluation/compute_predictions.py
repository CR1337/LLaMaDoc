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


SEEDS: List[int] = [42, 1337]
RANDOM_NUMBER_GENERATORS: List[Random] = [Random(s) for s in SEEDS]
SAMPLE_SIZE: int = 512
BATCH_SIZE: int = 32
GENERATION_PARAMETERS: GenerationParameters = GenerationParameters(
    max_new_tokens=256,
    sample_method=SampleMethod.TOP_P,
    top_p=0.85,
    temperature=0.5,
)
TEST_DATA_PATH: str = "evaluation/test_data.json"
PREDICTION_CACHE_PATHS: List[str] = [
    "cache/updated_docstrings.json",
    "cache/updated_docstrings_2.json"
]
try:
    with open(TEST_DATA_PATH, "r") as f:
        TEST_DATAS: List[List[Dict[str, Any]]] = [rng.sample(
            [
                {'c': item['c'], 'd': item['d'], 'l': item['l']}
                for item in json.load(f)
            ],
            k=SAMPLE_SIZE
        ) for rng in RANDOM_NUMBER_GENERATORS]
    BATCHED_TEST_DATAS: List[List[List[Dict[str, Any]]]] = [list(batched(d, BATCH_SIZE)) for d in TEST_DATAS]
except FileNotFoundError:
    TEST_DATA: List[List[Dict[str, Any]]] = []
    BATCHED_TEST_DATA: List[List[List[Dict[str, Any]]]] = []
    print("No test data found")

N_BATCHES: List[int] = [
    len(BATCHED_TEST_DATA[0]),
    len(BATCHED_TEST_DATA[1])
]


def perform_test(
    index: int,
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
        BATCHED_TEST_DATA[index], 
        total=N_BATCHES[index]
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


def compute_predictions(index: int) -> Dict[str, List[List[str]]]:
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
                    index=index,
                    mid=mid,
                    precaching=True,
                    test_parameters=test_parameters,
                    updated_docstrings=None
                )
            ]
            for _ in range(DistanceTest.N_SAMPLES)
        ]

    with open(PREDICTION_CACHE_PATHS[index], "w") as f:
        json.dump(results, f)
    