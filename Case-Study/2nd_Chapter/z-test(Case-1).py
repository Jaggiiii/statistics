import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate

# population varince known and variance equal

# nullhypo one with >co2 emission and the orhter with <co2 emission
def two_sample_z_test_case1(mean1, mean2, std_dev, size1, size2, alpha=0.05, tail="two"):
    standard_error = std_dev * math.sqrt((1 / size1) + (1 / size2))
    z_calculated = (mean1 - mean2) / standard_error
    if tail == "two":
        z_tabulated = stats.norm.ppf(1 - alpha / 2)
    elif tail in ["one_left", "one_right"]:
        z_tabulated = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError(f"Invalid tail type '{tail}'. Use 'one_left', 'one_right', or 'two'.")
    return z_calculated, z_tabulated, standard_error

def get_results_case1(mean1, mean2, std_dev, size1, size2, alpha=0.05, tail="two"):
    try:
        z_calculated, z_tabulated, standard_error = two_sample_z_test_case1(
            mean1, mean2, std_dev, size1, size2, alpha=alpha, tail=tail
        )
        if tail == "two":
            result = "Accept H0" if abs(z_calculated) <= z_tabulated else "Reject H0"
        elif tail == "one_left":
            result = "Accept H0" if z_calculated >= -z_tabulated else "Reject H0"
        elif tail == "one_right":
            result = "Accept H0" if z_calculated <= z_tabulated else "Reject H0"
    except ValueError as e:
        print(f"Error in two_sample_z_test_case1: {e}")
        return None, None
    
    results = [{
        'Mean 1': mean1,
        'Mean 2': mean2,
        'Std Dev (Common)': std_dev,
        'Size 1': size1,
        'Size 2': size2,
        'Alpha': alpha,
        'Tail': tail,
        'Z-calculated': f"{z_calculated:.4f}",
        'Z-tabulated': f"{z_tabulated:.4f}",
        'Result': result
    }]
    results_df = pd.DataFrame(results)
    print("\nTwo-Sample Z-Test Results (Case 1):")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

    if tail == "two":
        margin_of_error = z_tabulated * standard_error
        lower_bound = mean1 - mean2 - margin_of_error
        upper_bound = mean1 - mean2 + margin_of_error
        print(f"\nConfidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    else:
        if tail == "one_left":
            lower_bound = mean1 - mean2 - z_tabulated * standard_error
            upper_bound = None
            print(f"\nConfidence Interval (One Left-tailed): [{lower_bound:.4f}, ∞]")
        elif tail == "one_right":
            upper_bound = mean1 - mean2 + z_tabulated * standard_error
            lower_bound = None
            print(f"\nConfidence Interval (One Right-tailed): [-∞, {upper_bound:.4f}]")
    plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail)
    return lower_bound, upper_bound

def plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail):
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue')
    if tail == "two":
        plt.fill_between(x_values, y_values, where=(x_values <= -z_tabulated), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= z_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(-z_tabulated, color='black', linestyle='--', label=f'-Z Tabulated = {-z_tabulated:.2f}')
        plt.axvline(z_tabulated, color='black', linestyle='--', label=f'Z Tabulated = {z_tabulated:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
    elif tail == "one_left":
        plt.fill_between(x_values, y_values, where=(x_values <= -z_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(-z_tabulated, color='black', linestyle='--', label=f'-Z Tabulated = {-z_tabulated:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
    elif tail == "one_right":
        plt.fill_between(x_values, y_values, where=(x_values >= z_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(z_tabulated, color='black', linestyle='--', label=f'Z Tabulated = {z_tabulated:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
    plt.title(f'{tail.capitalize()}-tailed Z-Test: Hypothesis Testing Diagram')
    plt.xlabel('Z value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Load data and perform Z-test on Temperature
file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Convert Temperature column to numeric, dropping rows with NaN in Temperature
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df.dropna(subset=['Temperature'], inplace=True)

# Define the two groups based on CO2 Emissions (>=400 and <400)
group1 = df[df['CO2 Emissions'] >= 400]['Temperature'].head(40)
group2 = df[df['CO2 Emissions'] < 400]['Temperature'].head(45)

# Calculate means and common standard deviation
mean1 = group1.mean()
mean2 = group2.mean()
std_dev_common = math.sqrt(((len(group1) - 1) * group1.var(ddof=1) + (len(group2) - 1) * group2.var(ddof=1)) / (len(group1) + len(group2) - 2))

# Run Z-test
alpha = 0.05
tail = "two"  # Options: "one_left", "one_right", "two"
lower_bound, upper_bound = get_results_case1(mean1, mean2, std_dev_common, len(group1), len(group2), alpha=alpha, tail=tail)
