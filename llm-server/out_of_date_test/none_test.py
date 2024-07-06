from out_of_date_test.out_of_date_test import OutOfDateTest
from typing import List
from out_of_date_test.model import PredictionTestParameters, TestResult


class NoneTest(OutOfDateTest):
    
    def test(
        self, 
        codes: List[str],
        docstrings: List[str],
        parameters: PredictionTestParameters,
        generated_docstrings: List[str] | None
    ) -> List[TestResult]:
        assert len(codes) == len(docstrings)

        prompts = self._build_prompts(codes)
        if generated_docstrings is not None:
            updated_docstrings = [ds[0] for ds in generated_docstrings]
        else:
            updated_docstrings = self._get_updated_docstrings(prompts, parameters.generation_parameters)

        return [
            TestResult(
                updated_docstring=updated_docstring
            ) for updated_docstring in updated_docstrings  
        ]
    