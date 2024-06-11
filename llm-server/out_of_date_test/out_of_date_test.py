from abc import ABC, abstractmethod
import torch
from typing import List, Tuple, Generator
from out_of_date_test.model import TestParameters, TestResult, GenerationParameters
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from out_of_date_test.model_provider import ModelProvider, device


class OutOfDateTest(ABC):

    PREFIX_TOKEN: str = '<|fim_prefix|>'
    SUFFIX_TOKEN: str = '<|fim_suffix|>'
    MIDDLE_TOKEN: str = '<|fim_middle|>'
    FILE_SEPARATOR: str = '<|file_separator|>'
    INDENT: str = '    '
    HEADER_SEPARATOR: str = ':\n'

    PROMPT: str = f'{PREFIX_TOKEN}{{header}}\n{INDENT}"""{SUFFIX_TOKEN}"""\n{INDENT}{{body}}{MIDDLE_TOKEN}'

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
    
    def _split_codes(self, codes: List[str]) -> Generator[Tuple[str, str], None, None]:
        for code in codes:
            header, body = code.split(self.HEADER_SEPARATOR, 1)
            header = f"{header}{self.HEADER_SEPARATOR}"
            body = body.lstrip()
            yield header, body
    
    def _build_prompts(self, codes: List[str]) -> List[str]:
        return [
            self.PROMPT.format(header=header, body=body) 
            for header, body in self._split_codes(codes)
        ]
    
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
            docstring.split(self.MIDDLE_TOKEN)[-1].replace(self.FILE_SEPARATOR, '').strip()
            for docstring in updated_docstrings
        ]
        return updated_docstrings