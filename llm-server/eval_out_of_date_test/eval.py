import gc
import json
import logging
import os
import pickle
import sys
import warnings
from itertools import islice, product
from random import Random
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import transformers
from scipy.optimize import minimize
from tqdm import tqdm

from out_of_date_test.distance_test import DistanceTest
from out_of_date_test.model import (
    DistanceTestParameters, GenerationParameters,
    SampleMethod, DistanceFunction, NoneTestParameters, 
    TestParameters, TestResult
)
from out_of_date_test.model_provider import ModelProvider
from out_of_date_test.none_test import NoneTest


logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


SEED: int = 42
RANDOM_NUMBER_GENERATOR: Random = Random(SEED)
SAMPLE_SIZE: int = 512
BATCH_SIZE: int = 32
N_EXPLORATION_POINTS: int = 9
BOUNDS: List[Tuple[float, float]] = [(0.0, 0.2)]
EXPLORATION_POINTS: List[float] = np.linspace(BOUNDS[0][0], BOUNDS[0][1], N_EXPLORATION_POINTS).tolist()
PARAMETER_COMBINATIONS: List[Tuple[DistanceFunction, bool, bool]] = [
    (DistanceFunction.COSINE, False, False),
    (DistanceFunction.COSINE, False, True),
    (DistanceFunction.COSINE, True, False),
    (DistanceFunction.COSINE, True, True),
    (DistanceFunction.EUCLIDEAN, False, False),
    (DistanceFunction.EUCLIDEAN, False, True),
    (DistanceFunction.EUCLIDEAN, True, False),
    (DistanceFunction.EUCLIDEAN, True, True)
]
GENERATION_PARAMETERS: GenerationParameters = GenerationParameters(
    max_new_tokens=256,
    sample_method=SampleMethod.TOP_P,
    top_p=0.85,
    temperature=0.5,
)
TEST_DATA_PATH: str = "eval_out_of_date_test/test_data.json"
with open(TEST_DATA_PATH, "r") as f:
    TEST_DATA: List[Dict[str, Any]] = RANDOM_NUMBER_GENERATOR.sample(
        [
            {'c': item['c'], 'd': item['d'], 'l': item['l']}
            for item in json.load(f)
        ],
        k=SAMPLE_SIZE
    )
LABELS: List[bool] = [item['l'] for item in TEST_DATA]


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


BATCHED_TEST_DATA: List[List[Dict[str, Any]]] = list(batched(TEST_DATA, BATCH_SIZE))
N_BATCHES: int = len(BATCHED_TEST_DATA)
MAX_OPTIMIZATION_ITERATIONS: int = 64
XATOL: float = 1e-2
FATOL: float = 1e-2


def matthews_correlation(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator


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


def perform_test_for_evaluation(
    mid: str,
    distance_function: DistanceFunction,
    normalize: bool,
    sample_many: bool,
    test_threshold: float,
    updated_docstrings: Dict[str, List[str]],
    df: pd.DataFrame
) -> float:
    test_parameters = DistanceTestParameters(
        generation_parameters=GENERATION_PARAMETERS,
        distance_function=distance_function,
        normalize=normalize,
        sample_many=sample_many,
        test_threshold=test_threshold,
        mid=ModelProvider.embedding_model_ids[0]
    )
    results = perform_test(
        mid=mid,
        precaching=False,
        test_parameters=test_parameters,
        updated_docstrings=list(batched(updated_docstrings[mid], BATCH_SIZE))
    )
    tp, tn, fp, fn = 0, 0, 0, 0
    for result, label in zip(results, LABELS):
        classification = result.out_of_date
        if classification and label:
            tp += 1
        elif classification and not label:
            fp += 1
        elif not classification and label:
            fn += 1
        else:
            tn += 1
    mcc = matthews_correlation(tp, tn, fp, fn)
    df.loc[len(df)] = [
        mid, distance_function.value, normalize, sample_many, 
        test_threshold, tp, tn, fp, fn, mcc
    ]
    return mcc


def do_precaching() -> Dict[str, List[List[str]]]:
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
    return results


def do_evaluation_exploration(
    updated_docstrings: Dict[str, List[str]]
) -> pd.DataFrame:
    mids = list(reversed(ModelProvider.generative_model_ids))
    n_iterations = len(PARAMETER_COMBINATIONS) * len(mids) * N_EXPLORATION_POINTS

    df = pd.DataFrame(
        columns=[
            "mid", "distance_function", "normalize", "sample_many", 
            "test_threshold", "tp", "tn", "fp", "fn", "mcc"
        ]
    )
    
    for mid, (distance_function, normalize, sample_many), test_threshold in tqdm(
        product(mids, PARAMETER_COMBINATIONS, EXPLORATION_POINTS),
        total=n_iterations,
        desc="Exploration"
    ):
        perform_test_for_evaluation(
            mid=mid,
            distance_function=distance_function,
            normalize=normalize,
            sample_many=sample_many,
            test_threshold=test_threshold,
            updated_docstrings=updated_docstrings,
            df=df
        )
        
    return df


def objective_function(
    params: Tuple[float],
    updated_docstrings: Dict[str, List[str]],
    mid: str,
    distance_function: DistanceFunction,
    normalize: bool,
    sample_many: bool,
    df: pd.DataFrame
) -> float:
    test_threshold = params[0]
    mcc = perform_test_for_evaluation(
        mid=mid,
        distance_function=distance_function,
        normalize=normalize,
        sample_many=sample_many,
        test_threshold=test_threshold,
        updated_docstrings=updated_docstrings,
        df=df
    )
    return -mcc

    
class OptimizationCallback:

    def __init__(self, total: int):
        self._progress = tqdm(total=total, desc="Optimization")

    def __call__(self, _):
        self._progress.update(1)

    def close(self):
        self._progress.close()


def do_evaluation_optimization(
    updated_docstrings: Dict[str, List[str]],
    exploration_df: pd.DataFrame
) -> pd.DataFrame:
    mids = list(reversed(ModelProvider.generative_model_ids))
    n_iterations = len(mids) * len(PARAMETER_COMBINATIONS)

    df = pd.DataFrame(
        columns=[
            "mid", "distance_function", "normalize", "sample_many", 
            "test_threshold", "tp", "tn", "fp", "fn", "mcc"
        ]
    )

    for mid, (distance_function, normalize, sample_many) in tqdm(
        product(mids, PARAMETER_COMBINATIONS),
        total=n_iterations,
        desc="Exploration"
    ):
        initial_simplex = [tuple(exploration_df[
            (exploration_df["mid"] == mid) 
            & (exploration_df["distance_function"] == distance_function) 
            & (exploration_df["normalize"] == normalize) 
            & (exploration_df["sample_many"] == sample_many)
        ].nlargest(2, "mcc")["test_threshold"].to_list())]
        options = {
            "disp": False,
            "maxiter": MAX_OPTIMIZATION_ITERATIONS,
            "return_all": False,
            "initial_simplex": initial_simplex,
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": False,
            "bounds": BOUNDS
        }
        progress_callback = OptimizationCallback(MAX_OPTIMIZATION_ITERATIONS)
        minimize(
            objective_function,
            args=(
                updated_docstrings, 
                mid, distance_function, normalize, sample_many, 
                df
            ),
            method="Nelder-Mead",
            callback=progress_callback,
            options=options
        )

    return df


def load_or_compute(path: str, file_type: str, function: Callable, *args) -> Any:
    if os.path.exists(path):
        if file_type == 'df':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif file_type == 'json':
            with open(path, 'r') as f:
                return json.load(f)
    else:
        data = function(*args)
        if file_type == 'df':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif file_type == 'json':
            with open(path, 'w') as f:
                json.dump(data, f)
        os.chmod(path, 0o777)
        return data


def evaluation():
    updated_docstrings = load_or_compute(
        "cache/updated_docstrings.json",
        'json',
        do_precaching
    )
    exploration_df = load_or_compute(
        "cache/exploration_df.pkl",
        'df',
        do_evaluation_exploration,
        updated_docstrings
    )
    optimization_df = load_or_compute(
        "cache/optimization_df.pkl",
        'df',
        do_evaluation_optimization,
        updated_docstrings,
        exploration_df
    )
