from out_of_date_test.model import (
    PredictionTestParameters, GenerationParameters, 
    SampleMethod, TestCachingConfiguration
)
from out_of_date_test.prediction_test import PredictionTest
from out_of_date_test.model_provider import ModelProvider
from eval_out_of_date_test.eval import BATCH_SIZE

from typing import Dict, Generator, List
import json
from tqdm import tqdm
from eval_out_of_date_test.eval import batched
import torch
import gc


def test_parameter_generator(
    length: int, identifier: str, load: bool
) -> Generator[PredictionTestParameters, None, None]:
    for i in range(length):
        yield PredictionTestParameters(
            generation_parameters=GenerationParameters(
                max_new_tokens=256,
                sample_method=SampleMethod.TOP_P,
                top_p=0.85,
                temperature=0.5,
            ),
            caching_configuration=TestCachingConfiguration(
                cache_identifier=identifier,
                item_index=i,
                store=not load,
                load=load
            ),
            weight_decay=0.0,
            frequency_importance=0.0
        )


def get_test_data() -> List[Dict[str, str]]:
    with open("eval_out_of_date_test/test_data.json") as f:
        return [
            {'c': item['c'], 'd': item['d']} for item in json.load(f)
        ]
    

def run_tests():
    data = get_test_data()
    length = len(data) // BATCH_SIZE

    for mid in ModelProvider.generative_model_ids:
        batches = batched(data, BATCH_SIZE)
        for batch, parameters in tqdm(zip(
            batches, test_parameter_generator(length, f"test_data-{mid.replace('/', '-')}", load=False)
        ), total=length):
            test = PredictionTest(mid)
            codes = [item['c'] for item in batch]
            docstrings = [item['d'] for item in batch]
            test.test(codes, docstrings, parameters)

            del test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def do_precaching():
    run_tests()
