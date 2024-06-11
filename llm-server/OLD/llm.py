from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch
from typing import List
from huggingface_hub import login
from probability_computer import ProbabilityComputer
import os
from torch.nn.utils.rnn import pad_sequence

from model import TestQuery, TestResponse, GenerationParameters


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
    def _build_prompts(cls, codes: List[str]) -> List[str]:
        return [
            f"#<code>:\n{code}\n#<docstring>:\n"
            for code in codes
        ]

    @classmethod
    def _is_out_of_date(
        cls,
        docstring_probabilities: List[float],
        updated_docstring_probabilities: List[float],
        test_threshold: float
    ) -> List[bool]:
        return [
            (dp / udp) <= test_threshold
            for dp, udp in zip(
                docstring_probabilities, 
                updated_docstring_probabilities
            )
        ]

    @classmethod
    def update(cls, query: TestQuery) -> TestResponse:
        tokenizer = cls.tokenizers[query.mid]
        model = cls.models[query.mid]

        prompts = cls._build_prompts(query.codes)

        print("GENERATING UPDATED DOCSTRINGS")
        updated_docstrings = cls._get_updated_docstring(
            query.mid,
            query.codes,
            query.generation_parameters
        )

        probability_computer = ProbabilityComputer(
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        print("COMPUTING PROBABILITIES FOR USER DOCSTRINGS")
        docstring_probabilities = probability_computer.compute_docstring_probabilities(
            prompts, query.docstrings,
            query.test_parameters.weight_decay,
            query.test_parameters.frequency_importance
        )
        print("COMPUTING PROBABILITIES FOR GENERATED DOCSTRINGS")
        updated_docstring_probabilities = probability_computer.compute_docstring_probabilities(
            prompts, updated_docstrings,
            query.test_parameters.weight_decay,
            query.test_parameters.frequency_importance
        )
        print("PERFORMING TESTS")
        out_of_date = cls._is_out_of_date(
            docstring_probabilities, 
            updated_docstring_probabilities,
            query.test_parameters.test_threshold
        )

        print("CONSTRUCTING RESPONSE")
        return TestResponse(
            docstring_probabilities=docstring_probabilities,
            generated_docstring_probabilities=updated_docstring_probabilities,
            out_of_date=out_of_date,
            updated_docstrings=updated_docstrings
        )

    @classmethod
    def _get_updated_docstring(
        cls,
        model_id: str,
        codes: List[str],
        generation_parameters: GenerationParameters
    ) -> List[str]:
        tokenizer = cls.tokenizers[model_id]
        model = cls.models[model_id]

        prompts = cls._build_prompts(codes)

        print("TOKENIZING PROMPTS")
        # inputs = tokenizer(prompts, return_tensors='pt').to(device)
        inputs = tokenizer(prompts)['input_ids']
        single_input_tensors = [torch.tensor(inp) for inp in inputs]
        inputs_tensor = pad_sequence(single_input_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_masks = (inputs_tensor != tokenizer.pad_token_id).to(device)


        print("GENERATING DOCSTRINGS")
        # outputs = model.generate(
        #     # inputs.input_ids, 
        #     inputs_tensor,
        #     **(
        #         generation_parameters.to_torch_dict() 
        #         | {
        #             'eos_token_id': tokenizer.eos_token_id,
        #             'attention_mask': attention_masks
        #         }
        #     )
        # )

        print("DECODING DOCSTRINGS")
        # predictions = [
        #     tokenizer.decode(output)
        #     for output in outputs
        # ]

        # test request:
        """
        {
          "llm_id": "google/codegemma-2b",
          "codes": [
            "def double(x):\n    return 2 * x",
            "def swap(x, y):\n    return y, x"
          ],
          "docstrings": [
            "Triples the value x",
            "Swaps the values of x and y"
          ],
          "check_parameters": {
            "weight_decay": 0.5,
            "frequency_importance": 0.5,
            "test_threshold": 1
          },
          "generation_parameters": {
            "max_length": 64,
            "sample_method": "greedy"
          }
        }
        """

        # dummy prediction:
        dummy_prediction_0 = '<bos>#<code>:\ndef double(x):\n    return 2 * x\n#<docstring>:\n<pad>:\n    """\n    This function doubles the input x.\n    """\n    return x * 2\n<|file_separator|><eos>'
        dummy_prediction_1 = '<bos>#<code>:\ndef swap(x, y):\n    return y, x\n#<docstring>:\ndef swap(x, y):\n    return y, x\n<|file_separator|><eos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
        predictions = [dummy_prediction_0, dummy_prediction_1]

        print("PREDICTIONS:")
        for prediction in predictions:
            print(prediction)
            print()

        return predictions





























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

