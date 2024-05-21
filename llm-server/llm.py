from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch
from typing import List

from model import LlmQuery, LlmCheckResponse, LlmUpdateResponseResponse


gpu = torch.device('cuda:0')
model_id = 'google/codegemma-1.1-2b'
tokenizer = GemmaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=gpu, torch_dtype=torch.bfloat16)


def check(queries: List[LlmQuery]) -> List[LlmCheckResponse]:
    responses = []
    for query in queries:
        ...


def update(queries: List[LlmQuery]) -> List[LlmUpdateResponseResponse]:
    responses = []
    for query in queries:
        prompt = (
            f"<CODE>:\n{query.code}\n"
            f"<DOCSTRING>:\n{query.docstring}\n"
            f"<UPDATED DOCSTRING>:\n"
        )
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(gpu)
        outputs = model.generate(
            inputs,
            max_new_tokens=query.max_new_tokens,
            num_beams=query.num_beams,
            do_sample=query.do_sample,
            temperature=query.temperature,
            top_k=query.top_k,
            top_p=query.top_p,
            repetition_penalty=query.repetition_penalty,
            length_penalty=query.length_penalty,
        )
        answer = tokenizer.decode(outputs[0])
        responses.append(LlmUpdateResponseResponse(updated_docstring=answer))
    return responses


    