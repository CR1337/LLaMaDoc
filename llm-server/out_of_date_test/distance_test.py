from out_of_date_test.out_of_date_test import OutOfDateTest
from sentence_transformers import SentenceTransformer
from out_of_date_test.model import DistanceTestParameters, TestResult, DistanceFunction
from typing import List
import torch
from out_of_date_test.model_provider import device_name


class DistanceTest(OutOfDateTest):

    def test(
        self, 
        codes: List[str],
        docstrings: List[str],
        parameters: DistanceTestParameters
    ) -> List[TestResult]:
        assert len(codes) == len(docstrings)

        prompts = self._build_prompts(codes)
        updated_docstrings = self._get_updated_docstrings(prompts, parameters.generation_parameters)

        model = SentenceTransformer(parameters.mid, device=device_name)
        code_embeddings = model.encode(codes, normalize_embeddings=parameters.normalize)
        docstring_embeddings = model.encode(updated_docstrings, normalize_embeddings=parameters.normalize)
        updated_docstring_embeddings = model.encode(docstrings, normalize_embeddings=parameters.normalize)

        model.similarity_fn_name = parameters.distance_function.value

        docstring_similarities = model.similarity_pairwise(code_embeddings, docstring_embeddings)
        updated_docstring_similarities = model.similarity_pairwise(code_embeddings, updated_docstring_embeddings)

        if parameters.distance_function == DistanceFunction.EUCLIDEAN:
            docstring_similarities = -docstring_similarities
            updated_docstring_similarities = -updated_docstring_similarities

        ratios = torch.div(docstring_similarities, updated_docstring_similarities)
        results = [(ratio <= parameters.test_threshold).item() for ratio in ratios]

        test_results = [
            TestResult(
                out_of_date=out_of_date,
                updated_docstring=updated_docstring,
                docstring_score=docstring_score,
                updated_docstring_score=updated_docstring_score,
            ) for (
                out_of_date, updated_docstring, docstring_score, updated_docstring_score
            ) in zip(
                results, updated_docstrings, docstring_similarities, updated_docstring_similarities
            )
        ]

        return test_results
    