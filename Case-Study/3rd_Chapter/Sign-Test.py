import pandas as pd
import numpy as np
import scipy.stats as stats
import random

file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
data = pd.read_excel(file_path)

# Select the relevant column (for example, 'Temperature' in the dataset)
column_data = data['Temperature'].dropna().tolist()

# Select a random sample of 20 data points
random_sample = random.sample(column_data, 20)

# Calculate the conjectured median of the random sample
conjectured_median = sorted(random_sample)[len(random_sample) // 2]

def sign_test(data, conjectured_median, alpha=0.05):
    signs = []
    for value in data:
        if value > conjectured_median:
            signs.append('+')
        elif value < conjectured_median:
            signs.append('-')
        else:
            signs.append('0')

    n_positive = signs.count('+')
    n_negative = signs.count('-')
    sign_calculated_value = min(n_positive, n_negative)
    n = len(data)
    
    # Perform the sign test
    if n < 26:
        sign_tabulated_value = 4  # Using a fixed tabulated value for small sample sizes as an example
        decision = "Accept H0" if sign_calculated_value > sign_tabulated_value else "Reject H0"
    else:
        z_tabulated_value = stats.norm.ppf(1 - alpha / 2)
        z_calculated_value = abs(sign_calculated_value + 0.5 - (n / 2)) / np.sqrt(n / 2)
        decision = "Accept H0" if abs(z_calculated_value) <= z_tabulated_value else "Reject H0"

    # Print the results
    results_df = pd.DataFrame({'Data': data, 'Sign': signs})
    print("\nResults Table:")
    print(results_df)
    print(f"\nConjectured Median: {conjectured_median}")
    print(f"Number of Positive Signs: {n_positive}, Number of Negative Signs: {n_negative}")
    print(f"Sign Calculated Value: {sign_calculated_value}")
    if n >= 26:
        print(f"Z-Calculated Value: {z_calculated_value:.4f}")
        print(f"Z-Tabulated Value: {z_tabulated_value:.4f}")
    print(f"Decision: {decision}")

# Display the selected random sample
print("Selected Sample:", random_sample)

# Perform the sign test on the random sample of 'Temperature'
sign_test(random_sample, conjectured_median)
