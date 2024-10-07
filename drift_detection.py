import json.decoder
import numpy as np
import pandas as pd
import json

def scale_range(input, min, max):
        return (input - np.min(input)) / (np.max(input) - np.min(input)) * (max - min) + min

def calculate_psi(expected, actual, bins=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    """
    breakpoints = np.linspace(0, 1, bins + 1)

    expected_percents = np.histogram(scale_range(expected, 0, 1), bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(scale_range(actual, 0, 1), bins=breakpoints)[0] / len(actual)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

# Load datasets
old_data = pd.read_csv('old_data.csv')
new_data = pd.read_csv('new_data.csv')

# Compare distributions of important features
psi_values = {}
for col in old_data.columns:
    psi_values[col] = calculate_psi(old_data[col], new_data[col])

# Set a threshold for drift

drift_threshold = 0.2  # Example threshold

# Check if drift exceeds the threshold for any feature
drift_detected = any(psi > drift_threshold for psi in psi_values.values())

# Output drift results
if drift_detected:
    print("Drift detected. Retraining required.")
    exit(1)  # Exit with an error code to trigger retraining
else:
    print("No significant drift.")
    exit(0)
