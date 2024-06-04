import pickle

N = 256000
data = [1.0] * N

with open("token_frequencies.pkl", "wb") as file:
    pickle.dump(data, file)
