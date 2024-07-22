"""
This script generates the inverse relative log token frequencies from the training data.
The frequencies are not used by our current approach.
"""

import pickle
from math import log
import json
from transformers import AutoTokenizer
from out_of_date_test.model_provider import ModelProvider
from tqdm import tqdm

with open("../Jupyter/Train_Test_Data/train_data.json", 'r') as f:
    train_data = json.load(f)
docstrings = [e['d'] for e in train_data]

tokenizer = AutoTokenizer.from_pretrained(ModelProvider.generative_model_ids[0])
N = tokenizer.vocab_size
e = 1e-16

frequencies = [0 for _ in range(N)]
for docstring in tqdm(docstrings, total=len(docstrings)):
    tokens = tokenizer(docstring)['input_ids']
    for token in tokens:
        frequencies[token] += 1

print(f"{len(frequencies)=}")
print(f"{sum(frequencies)=}")

log_frequencies = [log(frequency + 1 + e) for frequency in frequencies]

print(f"{len(log_frequencies)=}")

max_log_frequency = max(log_frequencies)

relative_log_frequencies = [
    log_frequency / max_log_frequency
    for log_frequency in log_frequencies
]

print(f"{len(relative_log_frequencies)=}")

max_relative_log_frequency = max(relative_log_frequencies)

inverse_relative_log_frequencies = [
    # 1.0 / (relative_log_frequency + e)
    max_relative_log_frequency - relative_log_frequency 
    for relative_log_frequency in relative_log_frequencies
]

print(f"{len(inverse_relative_log_frequencies)=}")
print(f"{inverse_relative_log_frequencies=}")

with open("inverse_relative_log_token_frequencies.pkl", "wb") as file:
    pickle.dump(list(inverse_relative_log_frequencies), file)
