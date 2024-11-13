import pandas as pd
import numpy as np
from scipy import stats
import random

# Path to your dataset
file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"

# Read the dataset
data = pd.read_excel(file_path)

# Replace 'Mat' with the column name you want to use for analysis
# Here, I'm assuming you want to work with 'Temperature' and other variables like CO2 Emissions, Sea Level Rise, etc.
# I'll extract random samples from the Temperature and CO2 Emissions columns
temperature_data = data['Temperature'].dropna().tolist()
co2_data = data['CO2 Emissions'].dropna().tolist()

sample_size = 12

# Taking 2 * sample_size random samples (for before and after groups)
random_sample_temp = random.sample(temperature_data, 2 * sample_size)
random_sample_co2 = random.sample(co2_data, 2 * sample_size)

# Splitting into 'before' and 'after' groups
before_temp = random_sample_temp[:sample_size]
after_temp = random_sample_temp[sample_size:]
before_co2 = random_sample_co2[:sample_size]
after_co2 = random_sample_co2[sample_size:]

def wilcoxon_signed_rank_test(before, after, alpha=0.05, tail='two'):
    differences = np.array(before) - np.array(after)
    n = len(differences[differences != 0])  
    results_df = pd.DataFrame({
        'Before': before,
        'After': after,
        'Difference': differences
    })
    results_df['Absolute'] = np.abs(results_df['Difference'])
    results_df['Rank'] = results_df['Absolute'].rank(method='average')
    results_df['Signed Rank'] = np.where(results_df['Difference'] > 0, results_df['Rank'], 
                                         np.where(results_df['Difference'] < 0, -results_df['Rank'], 0))
    sum_positive_ranks = results_df[results_df['Difference'] > 0]['Rank'].sum()
    sum_negative_ranks = results_df[results_df['Difference'] < 0]['Rank'].sum()
    Ws = min(abs(sum_positive_ranks), abs(sum_negative_ranks))
    if n < 36:
        signed_tabulated = 14  
        decision = "Reject H0" if Ws <= signed_tabulated else "Accept H0"
        Z_Calculated = None  
        z_tabulated = signed_tabulated  
    else:
        mu_Ws = n * (n + 1) / 4
        sigma_Ws = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        Z_Calculated = (Ws - mu_Ws) / sigma_Ws
        if tail == 'two':
            z_tabulated = stats.norm.ppf(1 - alpha / 2)
            decision = "Reject H0" if abs(Z_Calculated) > z_tabulated else "Accept H0"
        elif tail == 'one_left':
            z_tabulated = stats.norm.ppf(1 - alpha)
            decision = "Reject H0" if Z_Calculated < -z_tabulated else "Accept H0"
        elif tail == 'one_right':
            z_tabulated = stats.norm.ppf(alpha)
            decision = "Reject H0" if Z_Calculated > z_tabulated else "Accept H0"
    print("\nResults Table for Wilcoxon signed rank test:")
    print(results_df)
    print(f"\nSum of positive ranks: {sum_positive_ranks}")
    print(f"Sum of negative ranks: {sum_negative_ranks}")
    print(f"Calculated Ws: {Ws}")
    if n >= 36:
        print(f"Mean (μ): {mu_Ws}, Standard Deviation (σ): {sigma_Ws}")
        print(f"Z Calculated Value: {Z_Calculated}")
        print(f"Z Tabulated Value: {z_tabulated}")
    print(f"Decision: {decision}")

# Print the samples selected
print("Selected Sample (Before) for Temperature:", before_temp)
print("Selected Sample (After) for Temperature:", after_temp)
print("Selected Sample (Before) for CO2 Emissions:", before_co2)
print("Selected Sample (After) for CO2 Emissions:", after_co2)

# Run the Wilcoxon Signed-Rank Test on Temperature
print("Wilcoxon Signed-Rank Test for Temperature Data:")
wilcoxon_signed_rank_test(before_temp, after_temp, tail='two')

# Run the Wilcoxon Signed-Rank Test on CO2 Emissions
print("Wilcoxon Signed-Rank Test for CO2 Emissions Data:")
wilcoxon_signed_rank_test(before_co2, after_co2, tail='two')
