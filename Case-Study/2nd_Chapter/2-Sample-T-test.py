import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from tabulate import tabulate
import scipy.stats as stats
import random

def load_sample_data(filename):
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook.active
    samples = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        try:
            co2_emissions = float(row[4])  # Change index to CO2 Emissions
            if co2_emissions > 0:
                samples.append(row)
        except (ValueError, TypeError):
            continue
    return samples

def calculate_pooled_variance(sample1, sample2):
    emissions1 = [float(player[4]) for player in sample1]  # Change index to CO2 Emissions
    emissions2 = [float(player[4]) for player in sample2]
    n1 = len(emissions1)
    n2 = len(emissions2)
    var1 = np.var(emissions1, ddof=1)
    var2 = np.var(emissions2, ddof=1)
    pooled_variance = ((n1 * var1 + n2 * var2) / (n1 + n2 - 2))
    return pooled_variance, var1, var2

def calculate_t(sample1, sample2, tail="two", alpha=0.05):
    emissions1 = [float(player[4]) for player in sample1]  # Change index to CO2 Emissions
    emissions2 = [float(player[4]) for player in sample2]
    n1 = len(emissions1)
    n2 = len(emissions2)
    
    if n1 == 0 or n2 == 0: 
        raise ValueError("Samples must contain data.")
        
    mean1 = np.mean(emissions1)
    mean2 = np.mean(emissions2)
    pooled_variance, var1, var2 = calculate_pooled_variance(sample1, sample2)
    
    t_calculated = (mean1 - mean2) / np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    t_tabulated = stats.t.ppf(1 - alpha / 2, df)
    
    return t_calculated, mean1, mean2, pooled_variance, var1, var2, t_tabulated, n1, n2, df

def plot_hypothesis_test(t_calculated, t_tabulated, alpha, tail, df):
    # Define the range of x values around the test statistic
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.t.pdf(x_values, df)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='T-distribution', color='blue')

    # Plot the critical regions
    if tail == "two":
        plt.fill_between(x_values, y_values, where=(x_values <= -t_tabulated), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= t_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(-t_tabulated, color='black', linestyle='--', label=f'-T Critical = {-t_tabulated:.2f}')
        plt.axvline(t_tabulated, color='black', linestyle='--', label=f'+T Critical = {t_tabulated:.2f}')
    elif tail == "one_left":
        plt.fill_between(x_values, y_values, where=(x_values <= -t_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(-t_tabulated, color='black', linestyle='--', label=f'-T Critical = {-t_tabulated:.2f}')
    elif tail == "one_right":
        plt.fill_between(x_values, y_values, where=(x_values >= t_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(t_tabulated, color='black', linestyle='--', label=f'+T Critical = {t_tabulated:.2f}')
    
    # Plot the calculated test statistic
    plt.axvline(t_calculated, color='red', linestyle='-', label=f'T Calculated = {t_calculated:.2f}')
    
    plt.xlabel('T Value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title("Hypothesis Test Plot for CO₂ Emissions T-Test")
    plt.show()

# Load your dataset and conduct the t-test
filename = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx" 
samples = load_sample_data(filename)
print(f"Total samples loaded: {len(samples)}")

# Create samples based on CO2 emissions thresholds
sample1 = random.sample([player for player in samples if float(player[4]) < 450], min(25, len(samples)))
sample2 = random.sample([player for player in samples if float(player[4]) >= 450], min(25, len(samples)))

n1 = len(sample1)
n2 = len(sample2)
print(f"Sample1 size (CO2 < 450): {n1}")
print(f"Sample2 size (CO2 >= 450): {n2}")
#H₀: The mean CO₂ emissions of the two groups are equal.

if n1 > 0 and n2 > 0:
    tail = "two"  # Can also be "one_left" or "one_right"
    alpha = 0.05
    t_calculated, mean1, mean2, pooled_variance, var1, var2, t_tabulated, n1, n2, df = calculate_t(sample1, sample2, tail=tail, alpha=alpha)
    
    # Confidence interval calculations
    standard_error = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))
    critical_value = stats.t.ppf(1 - alpha / 2, df)
    margin_of_error = critical_value * standard_error
    lower_bound = (mean1 - mean2) - margin_of_error
    upper_bound = (mean1 - mean2) + margin_of_error
    
    result = "Reject H0" if abs(t_calculated) > t_tabulated else "Accept H0"

    results = [{
        'Sample 1 Mean (CO2 < 450)': mean1,
        'Sample 2 Mean (CO2 >= 450)': mean2,
        'Pooled Variance': pooled_variance,
        'Variance1': var1,
        'Variance2': var2,
        'Calculated t': f"{t_calculated:.4f}",
        'Critical t': f"{t_tabulated:.4f}",
        'n1': n1,
        'n2': n2,
        'Confidence Interval': f"[{lower_bound:.4f}, {upper_bound:.4f}]",
        'Result': result
    }]
    
    print("\nT-Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    plot_hypothesis_test(t_calculated, t_tabulated, alpha=alpha, tail=tail, df=df)
else:
    print(f"Error: Sample sizes are n1 = {len(sample1)} and n2 = {len(sample2)}. Both must contain data for the t-test.")
