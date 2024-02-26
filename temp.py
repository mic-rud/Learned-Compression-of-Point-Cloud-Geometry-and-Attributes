import numpy as np
import matplotlib.pyplot as plt


quantized = np.load("temp/quantized_features.npy")[0]
unquantized = np.load("temp/unquantized_features.npy")[0]

variances = np.var(quantized, axis=1)
sorted_indices = np.argsort(variances)
quantized_all = quantized[sorted_indices]
unquantized_all = unquantized[sorted_indices]

# Load
for i in range(128):
