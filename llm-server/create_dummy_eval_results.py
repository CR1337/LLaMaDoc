# import pickle
# import pandas as pd
# import numpy as np
# from out_of_date_test.model import DistanceFunction

# # Constants
# MODEL_IDS = ["checkpoints/finetuned_0", "google/codegemma-2b"]
# PARAMETER_COMBINATIONS = [
#     (DistanceFunction.COSINE, False, False),
#     (DistanceFunction.COSINE, False, True),
#     (DistanceFunction.COSINE, True, False),
#     (DistanceFunction.COSINE, True, True),
#     (DistanceFunction.EUCLIDEAN, False, False),
#     (DistanceFunction.EUCLIDEAN, False, True),
#     (DistanceFunction.EUCLIDEAN, True, False),
#     (DistanceFunction.EUCLIDEAN, True, True)
# ]
# EXPLORATION_POINTS = np.linspace(0.0, 2.0, 9).tolist()
# OPTIMIZATION_STEPS = 10

# # Helper function to generate MCC values based on the assumptions
# def generate_mcc(distance_function, normalize, sample_many, test_threshold):
#     base_mcc = 0.5
#     if distance_function == DistanceFunction.EUCLIDEAN:
#         base_mcc += 0.1
#         if normalize:
#             base_mcc += 0.2
#     if sample_many:
#         base_mcc += 0.2
#     mcc = base_mcc - 0.05 * abs(test_threshold - 1)
#     return max(0, min(1, mcc))

# # Generate exploration data
# exploration_data = []

# for mid in MODEL_IDS:
#     for distance_function, normalize, sample_many in PARAMETER_COMBINATIONS:
#         for test_threshold in EXPLORATION_POINTS:
#             tp = np.random.randint(50, 100)
#             tn = np.random.randint(50, 100)
#             fp = np.random.randint(0, 50)
#             fn = np.random.randint(0, 50)
#             mcc = generate_mcc(distance_function, normalize, sample_many, test_threshold)
#             exploration_data.append([
#                 mid, distance_function.value, normalize, sample_many, 
#                 test_threshold, tp, tn, fp, fn, mcc
#             ])

# exploration_df = pd.DataFrame(
#     exploration_data,
#     columns=["mid", "distance_function", "normalize", "sample_many", 
#              "test_threshold", "tp", "tn", "fp", "fn", "mcc"]
# )

# # Generate optimization data with intermediate results
# optimization_data = []

# for mid in MODEL_IDS:
#     for distance_function, normalize, sample_many in PARAMETER_COMBINATIONS:
#         initial_threshold = 0.5 + np.random.rand()  # Start somewhere near 0.5 to 1.5
#         for step in range(OPTIMIZATION_STEPS):
#             test_threshold = initial_threshold + (np.random.rand() - 0.5) * 0.2  # Small random adjustments
#             tp = np.random.randint(50, 100)
#             tn = np.random.randint(50, 100)
#             fp = np.random.randint(0, 50)
#             fn = np.random.randint(0, 50)
#             mcc = generate_mcc(distance_function, normalize, sample_many, test_threshold)
#             optimization_data.append([
#                 mid, distance_function.value, normalize, sample_many, 
#                 test_threshold, tp, tn, fp, fn, mcc
#             ])

# optimization_df = pd.DataFrame(
#     optimization_data,
#     columns=["mid", "distance_function", "normalize", "sample_many", 
#              "test_threshold", "tp", "tn", "fp", "fn", "mcc"]
# )

# # Save data to files
# with open("cache/exploration_df_DUMMY.pkl", "wb") as f:
#     pickle.dump(exploration_df, f)
# with open("cache/optimization_df_DUMMY.pkl", "wb") as f:
#     pickle.dump(optimization_df, f)


import pickle
import pandas as pd
import numpy as np
from out_of_date_test.model import DistanceFunction

# Constants
MODEL_IDS = ["checkpoints/finetuned_0", "google/codegemma-2b"]
PARAMETER_COMBINATIONS = [
    (DistanceFunction.COSINE, False, False),
    (DistanceFunction.COSINE, False, True),
    (DistanceFunction.COSINE, True, False),
    (DistanceFunction.COSINE, True, True),
    (DistanceFunction.EUCLIDEAN, False, False),
    (DistanceFunction.EUCLIDEAN, False, True),
    (DistanceFunction.EUCLIDEAN, True, False),
    (DistanceFunction.EUCLIDEAN, True, True)
]
EXPLORATION_POINTS = np.linspace(0.0, 2.0, 9).tolist()
OPTIMIZATION_STEPS = 10

# Helper function to generate MCC values based on the assumptions
def generate_mcc(distance_function, normalize, sample_many, test_threshold):
    base_mcc = 0.5
    if distance_function == DistanceFunction.EUCLIDEAN:
        base_mcc += 0.1
        if normalize:
            base_mcc += 0.2
    if sample_many:
        base_mcc += 0.2
    mcc = base_mcc - 0.05 * abs(test_threshold - 1)
    mcc += np.random.normal(0, 0.01)  # Adding a small random noise to ensure no equal values
    return max(0, min(1, mcc))

# Generate exploration data
exploration_data = []

for mid in MODEL_IDS:
    for distance_function, normalize, sample_many in PARAMETER_COMBINATIONS:
        for test_threshold in EXPLORATION_POINTS:
            tp = np.random.randint(50, 100)
            tn = np.random.randint(50, 100)
            fp = np.random.randint(0, 50)
            fn = np.random.randint(0, 50)
            mcc = generate_mcc(distance_function, normalize, sample_many, test_threshold)
            exploration_data.append([
                mid, distance_function.value, normalize, sample_many, 
                test_threshold, tp, tn, fp, fn, mcc
            ])

exploration_df = pd.DataFrame(
    exploration_data,
    columns=["mid", "distance_function", "normalize", "sample_many", 
             "test_threshold", "tp", "tn", "fp", "fn", "mcc"]
)

# Generate optimization data with intermediate results
optimization_data = []

for mid in MODEL_IDS:
    for distance_function, normalize, sample_many in PARAMETER_COMBINATIONS:
        initial_threshold = 1.0 + np.random.normal(0, 0.1)  # Start near 1 with some noise
        for step in range(OPTIMIZATION_STEPS):
            adjustment = np.sin(step / OPTIMIZATION_STEPS * 2 * np.pi) * 0.2  # Sinusoidal adjustment
            test_threshold = initial_threshold + adjustment + np.random.normal(0, 0.01)  # Small noise
            tp = np.random.randint(50, 100)
            tn = np.random.randint(50, 100)
            fp = np.random.randint(0, 50)
            fn = np.random.randint(0, 50)
            mcc = generate_mcc(distance_function, normalize, sample_many, test_threshold)
            optimization_data.append([
                mid, distance_function.value, normalize, sample_many, 
                test_threshold, tp, tn, fp, fn, mcc
            ])

optimization_df = pd.DataFrame(
    optimization_data,
    columns=["mid", "distance_function", "normalize", "sample_many", 
             "test_threshold", "tp", "tn", "fp", "fn", "mcc"]
)

# Save data to files
with open("cache/exploration_df_DUMMY.pkl", "wb") as f:
    pickle.dump(exploration_df, f)
with open("cache/optimization_df_DUMMY.pkl", "wb") as f:
    pickle.dump(optimization_df, f)
