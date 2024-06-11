from abc import ABC, abstractmethod
import torch
from typing import List
from out_of_date_test.model import TestParameters, TestResult, GenerationParameters
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from out_of_date_test.model_provider import ModelProvider, device


class OutOfDateTest(ABC):

    CODE_PREFIX: str = "#<code>:\n"
    DOCSTRING_PREFIX: str = "#<docstring>:\n"
    PROMPT: str = f"{CODE_PREFIX}{{code}}\n{DOCSTRING_PREFIX}"

    _mid: str
    _device: torch.device
    _model: AutoModelForCausalLM
    _tokenizer: AutoTokenizer

    def __init__(self, mid: str):
        self._mid = mid
        self._model, self._tokenizer = ModelProvider.get_generative_model(mid)
        self._device = device

    @abstractmethod
    def test(
        self, 
        codes: List[str],
        docstrings: List[str],
        parameters: TestParameters
    ) -> List[TestResult]:
        raise NotImplementedError("@abstractmethod def test(...)")
    
    def _build_prompts(self, codes: List[str]) -> List[str]:
        return [self.PROMPT.format(code=code) for code in codes]
    
    def _get_updated_docstrings(self, prompts: List[str], parameters: GenerationParameters) ->List[str]:
        prompt_tokens = self._tokenizer(prompts)['input_ids']
        prompt_tokens_tensor = pad_sequence(
            [torch.tensor(tokens) for tokens in prompt_tokens], 
            batch_first=True, padding_value=self._tokenizer.pad_token_id
        ).to(self._device)
        prompt_attention_masks = (prompt_tokens_tensor != self._tokenizer.pad_token_id).to(self._device)

        params = parameters.to_torch_dict()
        params['eos_token_id'] = self._tokenizer.eos_token_id
        params['attention_mask'] = prompt_attention_masks
        updated_docstring_token_tensor = self._model.generate(prompt_tokens_tensor, **params)

        updated_docstrings = (
            self._tokenizer.decode(docstring_tokens, skip_special_tokens=True) 
            for docstring_tokens in updated_docstring_token_tensor
        )
        updated_docstrings = [
            docstring.split(self.DOCSTRING_PREFIX)[-1]
            for docstring in updated_docstrings
        ]
        return updated_docstrings