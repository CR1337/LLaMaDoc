from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch
from typing import List
from huggingface_hub import login
from probability_computer import ProbabilityComputer
import os

from model import LlmCheckQuery, LlmUpdateQuery, LlmCheckResponse, LlmUpdateResponse, CheckMethod


on_server: bool = not os.path.exists("not-on-server")
if on_server and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    with open("huggingface-token", "r") as file:
        token = file.read().strip()
    login(token)
    del token


class Llm:


    model_ids: List[str] = [
        'google/codegemma-2b'
    ]
    tokenizers = {
        model_id: GemmaTokenizer.from_pretrained(model_id)
        for model_id in model_ids
    }
    models = {
        model_id: AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        for model_id in model_ids
    }

    @classmethod
    def check(cls, query: LlmCheckQuery) -> LlmCheckResponse:
        tokenizer = cls.tokenizers[query.llm_id]
        model = cls.models[query.llm_id]

        prompts = [
            f"<|code|>{query.code}\n<|docstring|>"
            for query in query.codes
        ]

        probability_computer = ProbabilityComputer(
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        if query.check_method == CheckMethod.absolute:
            return LlmCheckResponse(
                probability_computer.compute_docstring_probabilities(
                    prompts, query.docstrings,
                    query.check_parameters.gamma,
                    query.check_parameters.use_weight_decay,
                    query.check_parameters.use_frequency_weights
                )
            )

        elif query.check_method == CheckMethod.relative:
            update_query = LlmUpdateQuery(
                llm_id=query.llm_id,
                codes=query.codes,
                generation_parameters=query.generation_parameters
            )
            update_response = cls.update(update_query)
            updated_docstrings = update_response.updated_docstrings
            return LlmCheckResponse(
                probability_computer.compute_docstring_probabilities(
                    prompts, query.docstrings,
                    query.check_parameters.gamma,
                    query.check_parameters.use_weight_decay,
                    query.check_parameters.use_frequency_weights
                ),
                probability_computer.compute_docstring_probabilities(
                    prompts, updated_docstrings,
                    query.check_parameters.gamma,
                    query.check_parameters.use_weight_decay,
                    query.check_parameters.use_frequency_weights
                )
            )

    @classmethod
    def update(cls, query: LlmUpdateQuery) -> LlmUpdateResponse:
        tokenizer = cls.tokenizers[query.llm_id]
        model = cls.models[query.llm_id]

        prompts = [
            f"<|code|>{code}\n<|docstring|>"
            for code in query.codes
        ]
        inputs = tokenizer(prompts, return_tensors='pt').to(device)

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





























# def _compute_weights(
#     probabilities: List[float], 
#     token_ids: List[int],
#     gamma: float, 
#     use_weight_decay: bool, 
#     use_frequency_weights: bool
# ) -> torch.Tensor:
#     num_probs = len(probabilities)

#     if use_weight_decay:
#         x = torch.linspace(0, 1, num_probs, device=probabilities.device).to(device)
#         weights = ((gamma + 1) / gamma) * (-x ** gamma + 1)
#     else:
#         weights = torch.ones(num_probs, device=probabilities.device).to(device)

#     if use_frequency_weights:
#         frequencies = torch.tensor(
#             [token_frequencies[token_id] for token_id in token_ids], 
#             device=probabilities.device
#         ).to(device)
#         weights *= frequencies

#     return weights


# def _compute_probabilities(
#     prompts: List[str], 
#     docstrings: List[str], 
#     tokenizer: GemmaTokenizer,
#     model: AutoModelForCausalLM,
#     gamma: float,
#     use_weight_decay: bool,
#     use_frequency_weights: bool
# ) -> List[float]:
#     docstrings = [
#         d + tokenizer.decode([tokenizer.eos_token_id])
#         for d in docstrings
#     ]

#     all_prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device)
#     all_docstring_tokens = tokenizer(docstrings, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device)

#     all_sequences = []
#     attention_masks = []

#     for prompt_token, docstring_token in zip(all_prompt_tokens, all_docstring_tokens):
#         sequences = []
#         masks = []
#         prompt_len = len(prompt_token)
#         continuation_len = len(docstring_token)

#         for i in range(continuation_len + 1):
#             combined = torch.cat((prompt_token, docstring_token[:i]), dim=0)
#             padding_len = prompt_len + continuation_len - len(combined)
#             padded_combined = torch.cat((combined, torch.full((padding_len,), tokenizer.pad_token_id, device=device)))
#             sequences.append(padded_combined)

#             mask = torch.cat((torch.ones(len(combined), device=device), torch.zeros(padding_len, device=device)))
#             masks.append(mask)

#         all_sequences.append(torch.stack(sequences))
#         attention_masks.append(torch.stack(masks))

#     all_sequences_tensor = torch.stack(all_sequences)
#     attention_masks_tensor = torch.stack(attention_masks)

#     n, sequence_len, token_len = all_sequences_tensor.shape
#     input_tensor = all_sequences_tensor.view(-1, token_len)
#     attention_mask_tensor = attention_masks_tensor.view(-1, token_len)

#     logits = model(input_tensor, attention_mask=attention_mask_tensor).logits.double()
#     logits = logits.view(n, sequence_len, token_len, -1)

#     last_token_logits = []
#     for i in range(n):
#         sequence_logits = []
#         for j in range(sequence_len):
#             # Find the position of the last actual token in the sequence
#             last_token_pos = int(attention_masks_tensor[i, j].sum().item() - 1)
#             sequence_logits.append(logits[i, j, last_token_pos])
#         last_token_logits.append(torch.stack(sequence_logits))

#     last_token_logits_tensor = torch.stack(last_token_logits)

#     softmax_tensor = torch.softmax(last_token_logits_tensor, dim=-1)

#     continuation_probs = []
#     all_token_ids = []

#     for i in range(n):
#         seq_probs = []
#         token_ids = []
#         for j in range(len(all_docstring_tokens[i])):
#             continuation_token_id = all_docstring_tokens[i][j].item()
#             token_prob = softmax_tensor[i, j + 1, continuation_token_id]
#             seq_probs.append(token_prob)
#             token_ids.append(continuation_token_id)
#         continuation_probs.append(seq_probs)
#         all_token_ids.append(token_ids)

#     continuation_prob_tensors = [torch.stack(seq) for seq in continuation_probs]

#     weighted_geom_means = []

#     for probs, token_ids in zip(continuation_prob_tensors, all_token_ids):
#         weights = _compute_weights(probs, token_ids, gamma, use_weight_decay, use_frequency_weights)
#         log_probs = probs.log()
#         weighted_log_probs = log_probs * weights
#         weighted_log_sum = weighted_log_probs.sum()
#         sum_weights = weights.sum()
#         weighted_geom_mean = torch.exp(weighted_log_sum / sum_weights)
#         weighted_geom_means.append(float(weighted_geom_mean))

#     return weighted_geom_means

