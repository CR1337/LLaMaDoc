from out_of_date_test.out_of_date_test import OutOfDateTest
from typing import List
from out_of_date_test.model import PredictionTestParameters, TestResult


class NoneTest(OutOfDateTest):
    
    def test(
        self, 
        codes: List[str],
        docstrings: List[str],
        parameters: PredictionTestParameters
    ) -> List[TestResult]:
        assert len(codes) == len(docstrings)

        prompts = self._build_prompts(codes)
        updated_docstrings = self._get_updated_docstrings(prompts, parameters.generation_parameters)

        if parameters.caching_configuration is not None:
            self._set_cache_file_permissions()

        return [
            TestResult(
                updated_docstring=updated_docstring
            ) for updated_docstring in updated_docstrings  
        ]
    