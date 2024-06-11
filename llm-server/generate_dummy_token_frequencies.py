import pickle
from math import log

N = 256000
e = 1e-16

frequencies = (1.0 for _ in range(N))
log_frequencies = (log(frequency) for frequency in frequencies)
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
