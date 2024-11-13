import math
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


   #This hypothesis assumes that there is no significant difference between the sample mean and the population mean.

def hypothesis_test(sample_mean, population_mean, std_dev, sample_size, alpha=0.10, tail="two"):
    # Calculate the standard error and degrees of freedom
    standard_error = std_dev / math.sqrt(sample_size - 1)
    df = sample_size - 1
    
    # Calculate the test statistic
    test_stat = (sample_mean - population_mean) / standard_error
    
    # Determine the critical value based on the tail type
    critical_value = stats.t.ppf(1 - alpha / 2, df) if tail == "two" else stats.t.ppf(1 - alpha, df)
    
    # Determine the hypothesis test result
    if tail == "two":
        result = "Accept H0" if abs(test_stat) <= critical_value else "Reject H0"
    elif tail == "one_left":
        result = "Accept H0" if test_stat >= -critical_value else "Reject H0"
    elif tail == "one_right":
        result = "Accept H0" if test_stat <= critical_value else "Reject H0"
    else:
        raise ValueError("Invalid tail type.")  
    
    # Prepare results for display
    results = [{
        'Sample Mean': sample_mean,
        'Population Mean': population_mean,
        'Std Dev': std_dev,
        'Sample Size': sample_size,
        'Alpha': alpha,
        'Tail': tail,
        'Test Statistic': f"{test_stat:.4f}",
        'Critical Value': f"{critical_value:.4f}",
        'Result': result
    }]
    
    results_df = pd.DataFrame(results)
    print("\n1-Sample T-test Results:")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Calculate and display the confidence interval
    if tail == "two":
        margin_of_error = critical_value * standard_error
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        print(f"\nConfidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    elif tail == "one_left":
        lower_bound = sample_mean - critical_value * standard_error
        upper_bound = None
        print(f"\nConfidence Interval (One Left-tailed): [{lower_bound:.4f}, ∞]")
    elif tail == "one_right":
        upper_bound = sample_mean + critical_value * standard_error
        lower_bound = None
        print(f"\nConfidence Interval (One Right-tailed): [-∞, {upper_bound:.4f}]")
    
    # Plot the results
    plot_hypothesis_test(test_stat, critical_value, alpha, tail)  
    return lower_bound, upper_bound

def plot_hypothesis_test(t_calculated, t_tabulated, alpha, tail):
    # Create a range of x values for the normal distribution
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue')
    
    # Plot the critical regions based on the tail type
    if tail == "two":
        t_tabulated_half = t_tabulated / 2
        plt.fill_between(x_values, y_values, where=(x_values <= -t_tabulated_half), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= t_tabulated_half), color='lightblue', alpha=0.5)
        plt.axvline(-t_tabulated_half, color='black', linestyle='--', label=f'-T Tabulated/2 = {-t_tabulated_half:.2f}')
        plt.axvline(t_tabulated_half, color='black', linestyle='--', label=f'+T Tabulated/2 = {t_tabulated_half:.2f}')  
        plt.axvline(-t_calculated, color='red', linestyle='-', label=f'-T Calculated = {-t_calculated:.2f}')
        plt.axvline(t_calculated, color='red', linestyle='-', label=f'+T Calculated = {t_calculated:.2f}')  
    elif tail == "one_left":
        plt.fill_between(x_values, y_values, where=(x_values <= -t_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(-t_tabulated, color='black', linestyle='--', label=f'-T Tabulated = {-t_tabulated:.2f}')
        plt.axvline(t_calculated, color='red', linestyle='-', label=f'T Calculated = {t_calculated:.2f}')
    elif tail == "one_right":
        plt.fill_between(x_values, y_values, where=(x_values >= t_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(t_tabulated, color='black', linestyle='--', label=f'T Tabulated = {t_tabulated:.2f}')
        plt.axvline(t_calculated, color='red', linestyle='-', label=f'T Calculated = {t_calculated:.2f}')
    
    plt.xlabel('Test Value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title("1-Sample T-test Results:")
    plt.show()

# Load the data
file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Sample from the 'Temperature' column instead of 'Wkts'
sample_data = df['Temperature'].sample(n=20)
sample_mean = sample_data.mean()

# Calculate the population mean and std deviation using 'Temperature'
population_mean = df['Temperature'].mean()
std_dev = df['Temperature'].std(ddof=0)

# Set sample size and other parameters
sample_size = 20
alpha = 0.10
tail = "two"

# Perform the hypothesis test
lower_bound, upper_bound = hypothesis_test(sample_mean, population_mean, std_dev, sample_size, alpha, tail)
