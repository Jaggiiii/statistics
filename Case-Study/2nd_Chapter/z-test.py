import math
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
# null hyp :The sample mean temperature is equal to the population mean temperature.
def z_test(sample_mean, population_mean, std_dev, sample_size, alpha=0.10, tail="two"):
    standard_error = std_dev / math.sqrt(sample_size)
    z_calculated = abs(sample_mean - population_mean) / standard_error
    if tail == "two":
        z_tabulated = stats.norm.ppf(1 - alpha / 2)
    elif tail in ["one_left", "one_right"]:
        z_tabulated = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError(f"Invalid tail type '{tail}'. Use 'one_left', 'one_right', or 'two'.")
    return z_calculated, z_tabulated, standard_error

def get_results(sample_mean, population_mean, std_dev, sample_size, alpha=0.10, tail="two"):
    try:
        z_calculated, z_tabulated, standard_error = z_test(
            sample_mean,
            population_mean,
            std_dev,
            sample_size,
            alpha=alpha,
            tail=tail
        )
        if tail == "two":
            result = "Accept H0" if abs(z_calculated) <= z_tabulated else "Reject H0"
        elif tail == "one_left":
            result = "Accept H0" if z_calculated >= -z_tabulated else "Reject H0"
        elif tail == "one_right":
            result = "Accept H0" if z_calculated <= z_tabulated else "Reject H0"
    except ValueError as e:
        print(f"Error in z_test: {e}")
        return None, None
    
    results = [{
        'Sample Mean': sample_mean,
        'Population Mean': population_mean,
        'Std Dev': std_dev,
        'Sample Size': sample_size,
        'Alpha': alpha,
        'Tail': tail,
        'Z-calculated': f"{z_calculated:.4f}",
        'Z-tabulated': f"{z_tabulated:.4f}",
        'Result': result
    }]
    
    results_df = pd.DataFrame(results)
    print("\nZ-Test Results:")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

    if tail == "two":
        margin_of_error = z_tabulated * standard_error
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        print(f"\nConfidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    else:
        if tail == "one_left":
            lower_bound = sample_mean - z_tabulated * standard_error
            upper_bound = None
            print(f"\nConfidence Interval (One Left-tailed): [{lower_bound:.4f}, ∞]")
        elif tail == "one_right":
            upper_bound = sample_mean + z_tabulated * standard_error
            lower_bound = None
            print(f"\nConfidence Interval (One Right-tailed): [-∞, {upper_bound:.4f}]")
    
    plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail)
    return lower_bound, upper_bound

def plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail):
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    z_calculated = abs(z_calculated)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue')
    
    if tail == "two":
        z_tabulated_half = z_tabulated 
        plt.fill_between(x_values, y_values, where=(x_values <= -z_tabulated_half), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= z_tabulated_half), color='lightblue', alpha=0.5)
        plt.axvline(-z_tabulated_half, color='black', linestyle='--', label=f'-Z Tabulated/2 = {-z_tabulated_half:.2f}')
        plt.axvline(z_tabulated_half, color='black', linestyle='--', label=f'+Z Tabulated/2 = {z_tabulated_half:.2f}')  
        plt.axvline(-z_calculated, color='red', linestyle='-', label=f'-Z Calculated/2 = {-z_calculated:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'+Z Calculated/2 = {z_calculated:.2f}')  
        plt.text(-z_tabulated_half - 0.5, -0.05, r'$Z_{tab}/2$', fontsize=14, color='black')
        plt.text(z_tabulated_half + 0.2, -0.05, r'$Z_{tab}/2$', fontsize=14, color='black')
        
        plt.text(-z_calculated - 0.5, -0.08, r'$Z_{cal}/2$', fontsize=14, color='red')
        plt.text(z_calculated + 0.2, -0.08, r'$Z_{cal}/2$', fontsize=14, color='red')
        
        plt.text(0, 0.2, 'Accept H0', fontsize=16, color='green', ha='center')
        plt.text(-3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
        plt.text(3.5, 0.02, 'Reject H0', fontsize=16, color='blue')
    elif tail == "one_left":
        plt.fill_between(x_values, y_values, where=(x_values <= -z_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(-z_tabulated, color='black', linestyle='--', label=f'-Z Tabulated = {-z_tabulated:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
        plt.text(1, 0.2, 'Accept H0', fontsize=16, color='green', ha='center')
        plt.text(-3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
    elif tail == "one_right":
        plt.fill_between(x_values, y_values, where=(x_values >= z_tabulated), color='lightblue', alpha=0.5)
        plt.axvline(z_tabulated, color='black', linestyle='--', label=f'Z Tabulated = {z_tabulated:.2f}')
        plt.axvline(z_calculated, color='red', linestyle='-', label=f'Z Calculated = {z_calculated:.2f}')
        plt.text(-1, 0.2, 'Accept H0', fontsize=16, color='green', ha='center')
        plt.text(3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
    
    plt.title(f'{tail.capitalize()}-tailed Z-Test: Hypothesis Testing Diagram')
    plt.xlabel('Z value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Replace the path with your actual Excel file path
file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Use temperature data instead
sample_data = df['Temperature'].sample(n=35)
sample_mean = sample_data.mean()
population_mean = df['Temperature'].mean()  # Adjusted for temperature data
std_dev = df['Temperature'].std(ddof=0)  # Adjusted for temperature data
sample_size = 35
alpha = 0.10
tail = "two"

lower_bound, upper_bound = get_results(sample_mean, population_mean, std_dev, sample_size, alpha, tail)
