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


def check(query: LlmCheckQuery) -> LlmCheckResponse:
    tokenizer = tokenizers[query.model_id]
    model = models[query.model_id]

    prompts = [
        f"<|code|>{query.code}\n<|docstring|>"
        for query in query.codes
    ]
    inputs = tokenizer(prompts, return_tensors='pt').to(gpu)

    if query.check_method == CheckMethod.absolute:
        ...

    elif query.check_method == CheckMethod.relative:
        ...

        
def update(query: LlmUpdateQuery) -> LlmUpdateResponse:
    tokenizer = tokenizers[query.model_id]
    model = models[query.model_id]

    prompts = [
        f"<|code|>{query.code}\n<|docstring|>"
        for query in query.codes
    ]
    inputs = tokenizer(prompts, return_tensors='pt').to(gpu)

    outputs = model.generate(
        inputs.input_ids, 
        **(
            query.generation_parameters.to_torch_dict() 
            | {'eos_token_id': tokenizer.eos_token_id}
        )
    )

    predictions = [
        tokenizer.decode(output)
        for output in outputs
    ]
    return LlmUpdateResponse(updated_docstrings=predictions)
    