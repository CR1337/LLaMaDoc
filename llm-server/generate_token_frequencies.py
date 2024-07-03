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

log_frequencies = (log(frequency + e) for frequency in frequencies)
relative_log_frequencies = (
    log_frequency / max(log_frequencies) 
    for log_frequency in log_frequencies
)
inverse_relative_log_frequencies = (
    1.0 / relative_log_frequency 
    for relative_log_frequency in relative_log_frequencies
)

with open("inverse_relative_log_token_frequencies.pkl", "wb") as file:
    pickle.dump(list(inverse_relative_log_frequencies), file)
