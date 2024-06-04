from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple
import pickle


class ProbabilityComputer:

    token_frequencies: List[float]
    with open("token_frequencies.pkl", "rb") as file:
        token_frequencies = pickle.load(file)

    _model: AutoModelForCausalLM
    _tokenizer: GemmaTokenizer
    _device: torch.device

    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: GemmaTokenizer,
        device: torch.device
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def compute_docstring_probabilities(
        self, 
        prompts: List[str],
        docstrings: List[str],
        gamma: float,
        use_weight_decay: bool,
        use_frequency_weights: bool
    ) -> List[float]:
        docstrings = self._append_eos_to_docstrings(docstrings)
        prompt_tokens = self._tokenize_texts(prompts)
        docstring_tokens = self._tokenize_texts(docstrings)

        all_sequences, attention_masks = self._prepare_sequences_and_masks(
            prompt_tokens, docstring_tokens
        )
        logits = self._compute_logits(all_sequences, attention_masks)
        last_token_logits = self._get_last_token_logits(logits, attention_masks)
        probabilities = torch.softmax(last_token_logits, dim=-1)
        last_docstring_token_probabilities, last_token_ids = self._compute_last_docstring_token_probabilities(
            probabilities, docstring_tokens
        )

        return self._compute_weighted_geometric_means(
            last_docstring_token_probabilities,
            last_token_ids,
            gamma,
            use_weight_decay,
            use_frequency_weights
        )

    def _append_eos_to_docstrings(self, docstrings: List[str]) -> List[str]:
        return [
            d + self._tokenizer.decode([self._tokenizer.eos_token_id])
            for d in docstrings
        ]
    
    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        return [
            self._tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )['input_ids'].to(self._device)
        ]
    
    def _prepare_sequences_and_masks(
        self,
        prompt_tokens: torch.Tensor, 
        docstring_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_sequences = []
        attention_masks = []

        for prompt_token, docstring_token in zip(prompt_tokens, docstring_tokens):
            sequences = []
            masks = []
            prompt_len = len(prompt_token)
            docstring_len = len(docstring_token)

            for i in range(docstring_len + 1):
                # Combine prompt and part of the docstring, progressively 
                # adding more of the docstring
                combined = torch.cat((prompt_token, docstring_token[:i]), dim=0)
                padding_len = prompt_len + docstring_len - len(combined)

                # Pad the combined sequence to the full length
                padded_combined = torch.cat((
                    combined, 
                    torch.full(
                        (padding_len,), 
                        self._tokenizer.pad_token_id, 
                        device=self._device
                    )
                ))
                sequences.append(padded_combined)

                # Create attention mask where the actual tokens are 1s and padding tokens are 0s
                mask = torch.cat((
                    torch.ones(len(combined), device=self._device), 
                    torch.zeros(padding_len, device=self._device)
                ))
                masks.append(mask)

            # Stack sequences and masks for the current prompt-docstring pair
            all_sequences.append(torch.stack(sequences))
            attention_masks.append(torch.stack(masks))

        return torch.stack(all_sequences), torch.stack(attention_masks)
    
    def _compute_logits(
        self,
        all_sequences: torch.Tensor, 
        attention_masks: torch.Tensor, 
    ) -> torch.Tensor:
        # Reshape tensors to fit the model input requirements
        n, sequence_len, token_len = all_sequences.shape
        input_tensor = all_sequences.view(-1, token_len)
        attention_masks = attention_masks.view(-1, token_len)

        # Compute logits from the model
        logits = self._model(
            input_tensor, attention_mask=attention_masks
        ).logits.double()
        return logits.view(n, sequence_len, token_len, -1)

    def _get_last_token_logits(
        self,
        logits: torch.Tensor, 
        attention_masks: torch.Tensor
    ) -> torch.Tensor:
        last_token_logits = []
        for i in range(logits.size(0)):
            sequence_logits = []
            for j in range(logits.size(1)):
                # Find the position of the last actual token in the sequence
                last_token_position = int(attention_masks[i, j].sum().item() - 1)
                sequence_logits.append(logits[i, j, last_token_position])
            last_token_logits.append(torch.stack(sequence_logits))

        return torch.stack(last_token_logits)
    
    def _compute_last_docstring_token_probabilities(
        self,
        softmax_tensor: torch.Tensor, 
        all_docstring_tokens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        all_docstring_probabilities = []
        all_token_ids = []

        for i in range(len(all_docstring_tokens)):
            docstring_probabilities = []
            token_ids = []
            for j in range(len(all_docstring_tokens[i])):
                docstring_token_id = all_docstring_tokens[i][j].item()
                # Extract the probability of the docstring token
                token_prob = softmax_tensor[i, j + 1, docstring_token_id]
                docstring_probabilities.append(token_prob)
                token_ids.append(docstring_token_id)
            all_docstring_probabilities.append(torch.stack(docstring_probabilities))
            all_token_ids.append(token_ids)

        return all_docstring_probabilities, all_token_ids
    
    def _compute_weighted_geometric_means(
        self,
        last_docstring_token_probabilities: List[torch.Tensor], 
        last_token_ids: List[List[int]], 
        gamma: float, 
        use_weight_decay: bool, 
        use_frequency_weights: bool
    ) -> List[float]:
        weighted_geom_means = []
    
        for probabilities, token_ids in zip(last_docstring_token_probabilities, last_token_ids):
            # Compute weights for the probabilities
            weights = self._compute_weights(
                probabilities, 
                token_ids, 
                gamma, 
                use_weight_decay, 
                use_frequency_weights
            )
            log_probs = probabilities.log()
            weighted_log_probs = log_probs * weights
            weighted_log_sum = weighted_log_probs.sum()
            sum_weights = weights.sum()
            # Calculate the weighted geometric mean
            weighted_geom_mean = torch.exp(weighted_log_sum / sum_weights)
            weighted_geom_means.append(float(weighted_geom_mean))
    
        return weighted_geom_means
    
    def _compute_weights(
        self,
        probabilities: List[float], 
        token_ids: List[int],
        gamma: float, 
        use_weight_decay: bool, 
        use_frequency_weights: bool
    ) -> torch.Tensor:
        num_probs = len(probabilities)

        if use_weight_decay:
            x = torch.linspace(0, 1, num_probs, device=probabilities.device).to(self._device)
            weights = ((gamma + 1) / gamma) * (-x ** gamma + 1)
        else:
            weights = torch.ones(num_probs, device=probabilities.device).to(self._device)

        if use_frequency_weights:
            frequencies = torch.tensor(
                [self.token_frequencies[token_id] for token_id in token_ids], 
                device=probabilities.device
            ).to(self._device)
            weights *= frequencies

        return weights
    