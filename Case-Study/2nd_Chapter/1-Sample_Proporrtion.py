import math
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

#  typically assumes that the population proportion is equal to a hypothesized value.

# IN THIS  CASE where we're testing whether the proportion of "high temperatures" (temperatures above 30°C) in the sample matches a hypothesized population proportion, the null hypothesis (H₀) would be:

# H₀: The proportion of high temperatures in the population is equal to the hypothesized proportion.

def hypothesis_test(sample_proportion, population_proportion, sample_size, alpha=0.10, tail="two"):
    # Calculate the standard error for proportions
    standard_error = math.sqrt((population_proportion * (1 - population_proportion)) / sample_size)
    
    # Calculate the test statistic (z-value)
    z_stat = (sample_proportion - population_proportion) / standard_error
    
    # Determine the critical value based on the tail type (using standard normal distribution)
    critical_value = stats.norm.ppf(1 - alpha / 2) if tail == "two" else stats.norm.ppf(1 - alpha)
    
    # Determine the hypothesis test result
    if tail == "two":
        result = "Accept H0" if abs(z_stat) <= critical_value else "Reject H0"
    elif tail == "one_left":
        result = "Accept H0" if z_stat >= -critical_value else "Reject H0"
    elif tail == "one_right":
        result = "Accept H0" if z_stat <= critical_value else "Reject H0"
    else:
        raise ValueError("Invalid tail type.")  
    
    # Prepare results for display
    results = [{
        'Sample Proportion': sample_proportion,
        'Population Proportion': population_proportion,
        'Sample Size': sample_size,
        'Alpha': alpha,
        'Tail': tail,
        'Test Statistic': f"{z_stat:.4f}",
        'Critical Value': f"{critical_value:.4f}",
        'Result': result
    }]
    
    results_df = pd.DataFrame(results)
    print("\n1-Sample Proportion Test Results:")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Calculate and display the confidence interval
    margin_of_error = critical_value * standard_error
    lower_bound = sample_proportion - margin_of_error
    upper_bound = sample_proportion + margin_of_error
    print(f"\nConfidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # Plot the results
    plot_hypothesis_test(z_stat, critical_value, alpha, tail)  
    return lower_bound, upper_bound

def plot_hypothesis_test(z_calculated, z_critical, alpha, tail):
    # Create a range of x values for the standard normal distribution
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue')
    
    # Plot the critical regions based on the tail type
    if tail == "two":
        plt.fill_between(x_values, y_values, where=(x_values <= -z_critical), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= z_critical), color='lightblue', alpha=0.5)
        plt.axvline(-z_critical, color='black', linestyle='--', label=f'-Z Critical = {-z_critical:.2f}')
        plt.axvline(z_critical, color='black', linestyle='--', label=f'+Z Critical = {z_critical:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
    elif tail == "one_left":
        plt.fill_between(x_values, y_values, where=(x_values <= -z_critical), color='lightblue', alpha=0.5)
        plt.axvline(-z_critical, color='black', linestyle='--', label=f'-Z Critical = {-z_critical:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
    elif tail == "one_right":
        plt.fill_between(x_values, y_values, where=(x_values >= z_critical), color='lightblue', alpha=0.5)
        plt.axvline(z_critical, color='black', linestyle='--', label=f'Z Critical = {z_critical:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
    
    plt.xlabel('Test Value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title("1-Sample Proportion Test Results:")
    plt.show()

# Load the data
file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Define a threshold for "high temperature" (e.g., 30°C)
threshold = 30

# Create a binary column where 1 represents temperatures greater than 30°C and 0 represents temperatures <= 30°C
df['High Temperature'] = (df['Temperature'] > threshold).astype(int)

# Sample from the 'High Temperature' column
sample_size = 20
sample_data = df['High Temperature'].sample(n=sample_size)

# Calculate the sample proportion of "high temperatures"
sample_proportion = sample_data.mean()

# Population proportion (e.g., 0.50, or 50% is the hypothesized proportion)
population_proportion = 0.50

# Perform the hypothesis test
lower_bound, upper_bound = hypothesis_test(sample_proportion, population_proportion, sample_size, alpha=0.10, tail="two")
