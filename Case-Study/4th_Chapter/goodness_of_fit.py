import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import re

# Path to the dataset
file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'
df = pd.read_excel(file_path)

# Let's use the first 10 rows for analysis as per your earlier request
df = df.head(10)

# You can select the 'Temperature' column or any other column you want to analyze
# If you'd like to apply transformations or handle NaNs for a specific column, let's assume you want to process the 'Temperature' column
df['Temperature'] = df['Temperature'].apply(lambda x: float(x) if pd.notnull(x) else np.nan)

# Drop rows where 'Temperature' column is NaN
df.dropna(subset=['Temperature'], inplace=True)

# Display data summary and basic statistics for 'Temperature'
print("Data Summary:")
print(df.head())
print("\nStatistics for 'Temperature':")
print(df['Temperature'].describe())

# Chi-Square Test Setup
chi_square_statistic = 5.0  # You can change this value based on your hypothesis or context
observed_frequencies = df['Temperature'].to_numpy()  # Using 'Temperature' as the observed data

# Expected frequencies could be the mean value repeated for the length of the observed frequencies
expected_frequencies = np.full_like(observed_frequencies, observed_frequencies.mean())

# Calculate degrees of freedom (number of observations - 1)
dof = len(observed_frequencies) - 1

# Calculate p-value using the Chi-Square cumulative distribution function (CDF)
p_value = 1 - chi2.cdf(chi_square_statistic, dof)

# Set significance level (alpha)
alpha = 0.05
chi_tab_value = chi2.ppf(1 - alpha, dof)

print(f"chi_tab_value:{chi_tab_value}")

# Display Chi-Square Test results
print("\nChi-Square Test Results:")
print(f"| {'Chi-Square Statistic':<20} | {'P-Value':<10} | {'Degrees of Freedom':<20} | {'Result':<10} |")
print(f"| {chi_square_statistic:<20.4f} | {p_value:<10.4e} | {dof:<20} | {'Reject H0' if p_value < 0.05 else 'Accept H0':<10} |")

# Plotting the Chi-Square distribution
x = np.linspace(0, chi_square_statistic + 10, 1000)
y = chi2.pdf(x, dof)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Chi-Square Distribution', color='blue')
plt.fill_between(x, 0, y, where=(x >= chi_square_statistic), color='red', alpha=0.5, label='Right Tail')
plt.axvline(chi_square_statistic, color='green', linestyle='--', label=f'Cutoff Value ({chi_square_statistic:.2f})')
plt.axvline(chi_tab_value, color='red', linestyle='--', label=f'Chi-Tab Value ({chi_tab_value:.2f})')

# Adding text to indicate the result and statistic values
result_text = 'Reject H0' if p_value < alpha else 'Accept H0'
plt.text(chi_square_statistic + 1, 0.01, f'{result_text}\nChi-Square: {chi_square_statistic:.2f}\nP-Value: {p_value:.4e}', 
         fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.text(chi_square_statistic + 1, 0.02, f'Chi-Square: {chi_square_statistic:.2f}, P-Value: {p_value:.4e}', 
         fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

# Labels and title
plt.xlabel('Chi-Square Value')
plt.ylabel('Probability Density')
plt.title('Right-Tailed Test Visualization for Chi-Square Distribution')
plt.legend()
plt.show()
