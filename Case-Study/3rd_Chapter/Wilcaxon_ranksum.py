import numpy as np
import pandas as pd
from scipy import stats
import random

file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
data = pd.read_excel(file_path)

# Example of selecting a relevant column, such as 'Temperature'
column_data = data['Temperature'].dropna().tolist()

# Randomly sample data for sample1 and sample2
sample1 = random.sample(column_data, 4)
sample2 = random.sample([x for x in column_data if x not in sample1], 7)

def wilcoxon_rank_sum_test(sample1, sample2, alpha=0.05, tail="two"):
    n1, n2 = len(sample1), len(sample2)
    z_tabulated = stats.norm.ppf(1 - alpha / (2 if tail == "two" else 1)) 
    combined_data = np.concatenate([sample1, sample2])
    ranks = stats.rankdata(combined_data)
    ranks_sample1 = ranks[:n1]
    ranks_sample2 = ranks[n1:]
    R = np.sum(ranks_sample1) if n1 < n2 else np.sum(ranks_sample2)
    mu_R = (min(n1, n2) * (n1 + n2 + 1)) / 2
    sigma_R = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    Z_Calculated = abs((R - mu_R) / sigma_R)
    
    if tail == "two":
        decision = "Reject H0" if abs(Z_Calculated) > z_tabulated else "Accept H0"
    elif tail == "one_left":
        decision = "Reject H0" if Z_Calculated < -z_tabulated else "Accept H0"
    elif tail == "one_right":
        decision = "Reject H0" if Z_Calculated > z_tabulated else "Accept H0"
    else:
        raise ValueError(f"Invalid tail type '{tail}'. Use 'one_left', 'one_right', or 'two'.")

    print("Results for Wilcoxon Rank Sum Test:")
    print(f"\nSample 1: {sample1}")
    print(f"Sample 2: {sample2}")
    print(f"Sum of ranks (R) for the smaller sample: {R}")
    print(f"Mean of Ranks (mu_R): {mu_R}")
    print(f"Standard Deviation of Ranks (sigma_R): {sigma_R}")
    print(f"Z Calculated Value: {Z_Calculated:.4f}")
    print(f"Z Tabulated Value: {z_tabulated:.4f}")
    print(f"Decision: {decision}")

# Call Wilcoxon Rank Sum Test
wilcoxon_rank_sum_test(sample1, sample2, tail="two")
