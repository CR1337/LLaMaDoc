from __future__ import annotations
from out_of_date_test.model import (
    PredictionTestParameters, DistanceTestParameters, GenerationParameters,
    SampleMethod, TestCachingConfiguration, DistanceFunction
)
from out_of_date_test.model_provider import ModelProvider
from out_of_date_test.prediction_test import PredictionTest
from out_of_date_test.distance_test import DistanceTest
from scipy.optimize import minimize
import itertools
import json
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import pickle
import os
import stat

from typing import List, Tuple


BATCH_SIZE: int = 32
test_data_batches = None


def get_generation_parameter() -> GenerationParameters:
    return GenerationParameters(
        max_new_tokens=256,
        sample_method=SampleMethod.TOP_P,
        top_p=0.85,
        temperature=0.5,
    )


def get_prediction_bounds() -> List[Tuple[int, int]]:
    return [(0, 2), (0, 1), (0, 1)]


def get_distance_bounds() -> List[Tuple[int, int]]:
    return [(0, 1)]


def get_distance_parameter_combinations() -> List[Tuple[DistanceFunction, bool, bool]]:
    return [
        (DistanceFunction.COSINE, False, False),
        (DistanceFunction.COSINE, False, True),
        (DistanceFunction.COSINE, True, False),
        (DistanceFunction.COSINE, True, True),
        (DistanceFunction.EUCLIDEAN, False, False),
        (DistanceFunction.EUCLIDEAN, False, True),
        (DistanceFunction.EUCLIDEAN, True, False),
        (DistanceFunction.EUCLIDEAN, True, True)
    ]


class ConfusionMatrix:
    
        def __init__(self):
            self.tp = 0
            self.tn = 0
            self.fp = 0
            self.fn = 0
    
        def update(self, classification: bool, label: bool):
            if classification and label:
                self.tp += 1
            elif not classification and not label:
                self.tn += 1
            elif classification and not label:
                self.fp += 1
            elif not classification and label:
                self.fn += 1
    
        def matthews_correlation_coefficient(self) -> float:
            return (
                (self.tp * self.tn - self.fp * self.fn)
                / (
                    (self.tp + self.fp) * (self.tp + self.fn) 
                    * (self.tn + self.fp) * (self.tn + self.fn)
                ) ** 0.5
            )
        

@dataclass
class IterationResult:

    parameters: Tuple[float, ...]
    score: float
    confusion_matrix: ConfusionMatrix
    mid: str

    @classmethod
    def results_to_dataframe(cls, results: List[IterationResult]) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=(
                [f"p_{i}" for i in range(len(results[0].parameters))] 
                + ["score", "tp", "tn", "fp", "fn", "mid", "is_final"]
            )
        )
        for i, result in enumerate(results):
            df.loc[len(df)] = [
                *result.parameters, 
                result.score, 
                result.confusion_matrix.tp, result.confusion_matrix.tn, 
                result.confusion_matrix.fp, result.confusion_matrix.fn, 
                result.mid,
                i == len(results) - 1
            ]
        return df



def prediction_objective_function(params: Tuple[float, float, float], mid: str, confusion_matrices: List[ConfusionMatrix]) -> float:
    threshold, weight_decay, frequency_importance = params

    confusion_matrix = ConfusionMatrix()

    for i, batch in enumerate(test_data_batches):
        codes = [item['c'] for item in batch]
        docstrings = [item['d'] for item in batch]
        labels = [item['l'] for item in batch]
        test_parameters = PredictionTestParameters(
            generation_parameters=get_generation_parameter(),
            caching_configuration=TestCachingConfiguration(
                cache_identifier=f"test-data_{mid}",
                item_index=i,
                store=False,
                load=True
            ),
            test_threshold=threshold,
            weight_decay=weight_decay,
            frequency_importance=frequency_importance
        )
        test = PredictionTest(mid)
        results = test.test(codes, docstrings, test_parameters)
        for result, label in zip(results, labels):
            classification = result.out_of_date
            confusion_matrix.update(classification, label)

    confusion_matrices.append(confusion_matrix)

    return -confusion_matrix.matthews_correlation_coefficient()


def distance_objective_function(params: Tuple[float], distance_function: DistanceFunction, normalize: bool, sample_many: bool, mid: str, confusion_matrices: List[ConfusionMatrix]) -> float:
    threshold = params[0]

    confusion_matrix = ConfusionMatrix()

    for i, batch in enumerate(test_data_batches):
        codes = [item['c'] for item in batch]
        docstrings = [item['d'] for item in batch]
        labels = [item['l'] for item in batch]
        test_parameters = DistanceTestParameters(
            generation_parameters=get_generation_parameter(),
            caching_configuration=TestCachingConfiguration(
                cache_identifier=f"test-data_{mid}",
                item_index=i,
                store=False,
                load=True
            ),
            mid=ModelProvider.embedding_model_ids[0],
            test_threshold=threshold,
            distance_function=distance_function,
            normalize=normalize,
            sample_many=sample_many
        )
        test = DistanceTest(mid)
        results = test.test(codes, docstrings, test_parameters)
        for result, label in zip(results, labels):
            classification = result.out_of_date
            confusion_matrix.update(classification, label)

    confusion_matrices.append(confusion_matrix)

    return -confusion_matrix.matthews_correlation_coefficient()


class ProgressCallback:

    def __init__(self, total: int):
        self._progress = tqdm(total=total)

    def __call__(self, _):
        self._progress.update(1)

    def close(self):
        self._progress.close()


def optimize_prediction() -> List[List[IterationResult]]:
    overall_results = []
    max_iter = 200 * 3
    initial_guess = [sum(vs) / 2 for vs in get_prediction_bounds()]
    options = {
        'return_all': True,
        'bounds': get_prediction_bounds(),
        'xatol': 1e-2,
        'fatol': 1e-2,
        'maxiter': max_iter
    }

    print("Starting prediction optimization...")
    n_iterations = len(ModelProvider.generative_model_ids)
    for i, mid in enumerate(ModelProvider.generative_model_ids):
        print(f"Optimizing for model {mid}... ({i + 1} / {n_iterations}")
        progress_callback = ProgressCallback(max_iter)
        confusion_matrices = []
        results = minimize(
            prediction_objective_function,
            initial_guess,
            args=(mid, confusion_matrices),
            method="Nelder-Mead",
            callback=progress_callback,
            options=options
        )
        progress_callback.close()
        overall_results.append([IterationResult(r.x, r.fun, confusion_matrix, mid) for r, confusion_matrix in zip(results, confusion_matrices)])


    return overall_results


def optimize_distance() -> List[List[IterationResult]]:
    overall_results = []
    max_iter = 200
    initial_guess = [sum(vs) / 2 for vs in get_distance_bounds()]
    options = {
        'return_all': True,
        'bounds': get_distance_bounds(),
        'xatol': 1e-2,
        'fatol': 1e-2,
        'maxiter': max_iter  
    }

    print("Starting distance optimization...")
    n_iterations = len(ModelProvider.generative_model_ids) * len(get_distance_parameter_combinations())
    for i, ((distance_function, normalize, sample_many), mid) in enumerate(itertools.product(get_distance_parameter_combinations(), ModelProvider.generative_model_ids)):
        print(f"Optimizing for model {mid} with distance function {distance_function}, normalize={normalize}, sample_many={sample_many}... ({i + 1} / {n_iterations})")
        progress_callback = ProgressCallback(max_iter)
        confusion_matrices = []
        results = minimize(
            distance_objective_function,
            initial_guess,
            args=(distance_function, normalize, sample_many, mid, confusion_matrices),
            method="Nelder-Mead",
            callback=progress_callback,
            options=options
        )
        progress_callback.close()
        overall_results.append([IterationResult(r.x, r.fun, confusion_matrix, mid) for r, confusion_matrix in zip(results, confusion_matrices)])

    return overall_results


def optimize_parameters() -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_results = optimize_prediction()
    distance_results = optimize_distance()

    predictions_dfs = (IterationResult.results_to_dataframe(results) for results in prediction_results)
    distances_dfs = (IterationResult.results_to_dataframe(results) for results in distance_results)

    prediction_df = pd.concat(predictions_dfs)
    distance_df = pd.concat(distances_dfs)

    return prediction_df, distance_df


def store_results(prediction_df: pd.DataFrame, distance_df: pd.DataFrame):
    with open("cache/prediction_eval_results.pkl", "wb") as f:
        pickle.dump(prediction_df, f)
    with open("cache/distance_eval_results.pkl", "wb") as f:
        pickle.dump(distance_df, f)

    os.chmod("cache/prediction_eval_results.pkl", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod("cache/distance_eval_results.pkl", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def do_eval():
    with open("eval_out_of_date_test/test_data.json") as f:
        test_data = [
            {'c': item['c'], 'd': item['d'], 'l': item['l']} 
            for item in json.load(f)
        ]
    global test_data_batches
    test_data_batches = itertools.batched(test_data, BATCH_SIZE)

    prediction_df, distance_df = optimize_parameters()
    store_results(prediction_df, distance_df)
