from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch
from typing import List

from model import LlmCheckQuery, LlmUpdateQuery, LlmCheckResponse, LlmUpdateResponse, SampleMethod, CheckMethod


gpu = torch.device('cuda:0')

model_ids: List[str] = [
    'google/codegemma-1.1-2b',
    'google/codegemma-1.1-7b'
]
tokenizers = {
    model_id: GemmaTokenizer.from_pretrained(model_id)
    for model_id in model_ids
}
models = {
    model_id: AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=gpu,
        torch_dtype=torch.bfloat16
    )
    for model_id in model_ids
}


def _check_absolute(tokenizer: GemmaTokenizer, model: AutoModelForCausalLM, code: str, docstring: str) -> LlmCheckResponse:
    """
    Determine the probability of a docstring given a code snippet.
    Therefore calculate the probability for the next token after the code snippet. Append the token
    top the prompt and repeat the process until the end of the eos token is reached.
    Then return the geometric mean of the probabilities.
    """

    prompt = (
        f"<|code|>{code}\n"
        f"<|docstring|>"
    )



def _check_relative(tokenizer: GemmaTokenizer, model: AutoModelForCausalLM, code: str, docstring: str) -> LlmCheckResponse:
    ...


def check(queries: List[LlmCheckQuery]) -> List[LlmCheckResponse]:
    responses = []

    for query in queries:
        tokenizer = tokenizers[query.model_id]
        model = models[query.model_id]

        if query.check_method == CheckMethod.absolute:
            responses.append(_check_absolute(tokenizer, model, query.code, query.docstring))
        elif query.check_method == CheckMethod.relative:
            responses.append(_check_relative(tokenizer, model, query.code, query.docstring))

    return responses

        
def update(queries: List[LlmUpdateQuery]) -> List[LlmUpdateResponse]:
    responses = []

    for query in queries:
        tokenizer = tokenizers[query.model_id]
        model = models[query.model_id]

        prompt = (
            f"<|code|>{query.code}\n"
            f"<|docstring|>"
        )
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(gpu)

        generation_parameters = {
            'max_length': query.max_length,
            'temperature': query.temperature,
            'repetition_penalty': query.repetition_penalty,
            'length_penalty': query.length_penalty,
            'eos_token_id': tokenizer.eos_token_id
        }

        if query.sample_method == SampleMethod.greedy:
            pass
        elif query.sample_method == SampleMethod.top_k:
            generation_parameters['top_k'] = query.top_k
        elif query.sample_method == SampleMethod.top_p:
            generation_parameters['top_p'] = query.top_p
        elif query.sample_method == SampleMethod.beam:
            generation_parameters['num_beams'] = query.num_beams
            generation_parameters['early_stopping'] = query.early_stopping

        outputs = model.generate(inputs, **generation_parameters)
        answer = tokenizer.decode(outputs[0])

        responses.append(LlmUpdateResponse(updated_docstring=answer))
    
    return responses


    