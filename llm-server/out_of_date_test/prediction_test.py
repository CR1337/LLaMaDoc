from out_of_date_test.out_of_date_test import OutOfDateTest
from typing import List, Tuple
from out_of_date_test.model import PredictionTestParameters, TestResult
import torch
from math import tan, pi
import pickle
import gc


class PredictionTest(OutOfDateTest):

    inverse_relative_log_token_frequencies: List[float]
    with open("out_of_date_test/inverse_relative_log_token_frequencies.pkl", "rb") as file:
        inverse_relative_log_token_frequencies = pickle.load(file)

    def test(
        self, 
        codes: List[str],
        docstrings: List[str],
        parameters: PredictionTestParameters,
        generated_docstrings: List[str] | None = None
    ) -> List[TestResult]:
        assert len(codes) == len(docstrings)

        prompts = self._build_prompts(codes)
        if generated_docstrings is not None:
            updated_docstrings = [ds[0] for ds in generated_docstrings]
        else:
            updated_docstrings = self._get_updated_docstrings(prompts, parameters.generation_parameters)

        docstring_probabilities = self.compute_docstring_probabilities(
            prompts, docstrings,
            parameters.weight_decay, parameters.frequency_importance
        )

        updated_docstring_probabilities = self.compute_docstring_probabilities(
            prompts, updated_docstrings,
            parameters.weight_decay, parameters.frequency_importance
        )

        is_out_of_date = self._is_out_of_date(
            docstring_probabilities, updated_docstring_probabilities,
            parameters.test_threshold
        )

        return [
            TestResult(
                out_of_date=out_of_date,
                updated_docstring=updated_docstring,
                docstring_score=docstring_score,
                updated_docstring_score=updated_docstring_score
            ) for (
                out_of_date, 
                updated_docstring, 
                docstring_score, 
                updated_docstring_score
            ) in zip(
                is_out_of_date, 
                updated_docstrings, 
                docstring_probabilities, 
                updated_docstring_probabilities
            )
        ]
    
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
    
    def compute_docstring_probabilities(
        self, 
        prompts: List[str],
        docstrings: List[str],
        weight_decay: float,
        frequency_importance: float
    ) -> List[float]:
        docstrings = self._append_eos_to_docstrings(docstrings)
        prompt_tokens = self._tokenize_texts(prompts)
        docstring_tokens = self._tokenize_texts(docstrings)

        all_sequences, attention_masks = self._prepare_sequences_and_masks(
            prompt_tokens, docstring_tokens
        )
        probability_distributions = self._compute_probability_distributions(all_sequences, attention_masks)
        last_token_probability_distributions = self._get_last_token_probability_distributions(probability_distributions, attention_masks)
        last_docstring_token_probabilities, last_token_ids = self._compute_last_docstring_token_probabilities(
            last_token_probability_distributions, docstring_tokens
        )

        clamp = lambda value, min_value, max_value: max(min(value, max_value), min_value)

        epsilon = 1e-16
        decay_exponent = tan(
            (1.0 - clamp(weight_decay, epsilon, 1.0 - epsilon)) * pi / 2
        )

        result = self._compute_weighted_geometric_means(
            last_docstring_token_probabilities,
            last_token_ids,
            decay_exponent,
            frequency_importance
        )

        del probability_distributions, last_token_probability_distributions, last_docstring_token_probabilities
        gc.collect()
        torch.cuda.empty_cache()

        return result
    
    def _append_eos_to_docstrings(self, docstrings: List[str]) -> List[str]:
        return [
            d + self._tokenizer.decode([self._tokenizer.eos_token_id])
            for d in docstrings
        ]
    
    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        return self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )['input_ids'].to(self._device)
    
    def _prepare_sequences_and_masks(
        self,
        all_prompt_tokens: List[torch.Tensor], 
        all_docstring_tokens: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_sequences = []
        attention_masks = []

        for prompt_tokens, docstring_tokens in zip(all_prompt_tokens, all_docstring_tokens):
            sequences = []
            masks = []
            prompt_len = len(prompt_tokens)
            docstring_len = len(docstring_tokens)

            for i in range(docstring_len + 1):
                # Combine prompt and part of the docstring, progressively 
                # adding more of the docstring
                combined = torch.cat((prompt_tokens, docstring_tokens[:i]), dim=0)
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
        
        del combined, padded_combined, sequences, masks
        gc.collect()
        torch.cuda.empty_cache()

        return torch.stack(all_sequences), torch.stack(attention_masks)
    
    def _compute_probability_distributions(
        self,
        all_sequences: torch.Tensor, 
        attention_masks: torch.Tensor
    ) -> torch.Tensor:
        # Reshape tensors to fit the model input requirements
        n_batches, n_sequences, n_tokens = all_sequences.shape
        input_tensor = all_sequences.view(-1, n_tokens)
        attention_masks = attention_masks.view(-1, n_tokens)

        # Compute logits from the model
        logits = self._model(
            input_tensor, attention_mask=attention_masks
        ).logits.double()
        # logits = None
        probabilities = torch.softmax(logits, dim=-1)

        del input_tensor, attention_masks, logits
        gc.collect()
        torch.cuda.empty_cache()

        result = probabilities.view(n_batches, n_sequences, n_tokens, -1)
        return result

    def _get_last_token_probability_distributions(
        self,
        probability_distributions: torch.Tensor, 
        attention_masks: torch.Tensor
    ) -> torch.Tensor:
        last_token_probability_distributions = []
        for i in range(probability_distributions.size(0)):
            sequence_probability_distributions = []
            for j in range(probability_distributions.size(1)):
                # Find the position of the last actual token in the sequence
                last_token_position = int(attention_masks[i, j].sum().item() - 1)
                sequence_probability_distributions.append(probability_distributions[i, j, last_token_position])
            last_token_probability_distributions.append(torch.stack(sequence_probability_distributions))

        return torch.stack(last_token_probability_distributions)
    
    def _compute_last_docstring_token_probabilities(
        self,
        probability_distributions: torch.Tensor, 
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
                token_prob = probability_distributions[i, j + 1, docstring_token_id]
                docstring_probabilities.append(token_prob)
                token_ids.append(docstring_token_id)
            all_docstring_probabilities.append(torch.stack(docstring_probabilities))
            all_token_ids.append(token_ids)

        return all_docstring_probabilities, all_token_ids

    def _compute_weighted_geometric_means(
        self,
        last_docstring_token_probabilities: List[torch.Tensor], 
        last_token_ids: List[List[int]], 
        decay_exponent: float, 
        frequency_importance: float
    ) -> List[float]:
        weighted_geom_means = []
    
        for probabilities, token_ids in zip(last_docstring_token_probabilities, last_token_ids):
            # Compute weights for the probabilities
            weights = self._compute_weights(
                probabilities, 
                token_ids, 
                decay_exponent,
                frequency_importance
            )
            log_probs = probabilities.log()
            weighted_log_probs = log_probs * weights
            weighted_log_sum = weighted_log_probs.sum()
            # Calculate the weighted geometric mean
            weighted_geom_mean = torch.exp(weighted_log_sum / weights.sum())
            weighted_geom_means.append(float(weighted_geom_mean))

        del weights, log_probs, weighted_log_probs, weighted_log_sum, weighted_geom_mean
        gc.collect()
        torch.cuda.empty_cache()
    
        return weighted_geom_means
    
    def _compute_weights(
        self,
        probabilities: List[float], 
        token_ids: List[int],
        decay_exponent: float, 
        frequency_importance: float
    ) -> torch.Tensor:
        frequencies = torch.tensor(
            [
                self.inverse_relative_log_token_frequencies[token_id] 
                for token_id in token_ids
            ]
        ).to(self._device)
        x = torch.linspace(0, 1, len(probabilities)).to(self._device)
        decay_weights = 1 - x ** decay_exponent
        frequency_weights = torch.lerp(
            torch.ones_like(decay_weights),
            frequencies,
            frequency_importance
        )
        weights = decay_weights * frequency_weights
        weights /= weights.sum()

        del frequencies, x, decay_weights, frequency_weights
        gc.collect()
        torch.cuda.empty_cache()

        return weights
