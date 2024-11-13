import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate
import scipy.stats as stats

# population varince unknown and variance equal
 #There is no significant difference between the average temperatures of the two groups.
def load_sample_data(filename):
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook.active
    samples = []    
    for row in sheet.iter_rows(min_row=2, values_only=True):
        try:
            temperature = float(row[3])  # Adjusted for Temperature column
            if temperature > 0: 
                samples.append(row)  
        except (ValueError, TypeError):
            continue
    return samples

def calculate_pooled_variance(sample1, sample2):
    temperature1 = [float(player[3]) for player in sample1]  # Adjusted for Temperature column
    temperature2 = [float(player[3]) for player in sample2]  # Adjusted for Temperature column
    n1 = len(temperature1)
    n2 = len(temperature2)
    var1 = np.var(temperature1, ddof=1)
    var2 = np.var(temperature2, ddof=1)
    pooled_variance = ((n1 * var1**2 + n2 * var2**2) / (n1 + n2 - 2))
    return pooled_variance, var1, var2

def calculate_z(sample1, sample2, tail="two", alpha=0.05):
    temperature1 = [float(player[3]) for player in sample1]  # Adjusted for Temperature column
    temperature2 = [float(player[3]) for player in sample2]  # Adjusted for Temperature column
    n1 = len(temperature1)
    n2 = len(temperature2)
    if n1 == 0 or n2 == 0: 
        raise ValueError("Samples must contain data.")
    mean1 = np.mean(temperature1)
    mean2 = np.mean(temperature2)
    pooled_variance, var1, var2 = calculate_pooled_variance(sample1, sample2)
    z_calculated = (mean1 - mean2) / np.sqrt(pooled_variance * (1 / (n1 - 1) + 1 / (n2 - 1)))
    if tail == "two":
        critical_value = norm.ppf(1 - alpha / 2)
    else:
        critical_value = norm.ppf(1 - alpha)
    return z_calculated, mean1, mean2, pooled_variance, var1, var2, critical_value

def plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail):
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue')
    if tail == "two":
        z_tabulated_half = z_tabulated / 2
        z_calculated_half = z_calculated / 2
        plt.fill_between(x_values, y_values, where=(x_values <= -z_tabulated_half), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= z_tabulated_half), color='lightblue', alpha=0.5)
        plt.axvline(-z_tabulated_half, color='black', linestyle='--', label=f'-Z Tabulated/2 = {-z_tabulated_half:.2f}')
        plt.axvline(z_tabulated_half, color='black', linestyle='--', label=f'+Z Tabulated/2 = {z_tabulated_half:.2f}')  
        plt.axvline(-z_calculated_half, color='red', linestyle='-', label=f'-Z Calculated/2 = {-z_calculated_half:.2f}')
        plt.axvline(z_calculated_half, color='red', linestyle='-', label=f'+Z Calculated/2 = {z_calculated_half:.2f}')  
        plt.text(-z_tabulated_half - 0.5, -0.05, r'$Z_{tab}/2$', fontsize=14, color='black')
        plt.text(z_tabulated_half + 0.2, -0.05, r'$Z_{tab}/2$', fontsize=14, color='black')
        
        plt.text(-z_calculated_half - 0.5, -0.08, r'$Z_{cal}/2$', fontsize=14, color='red')
        plt.text(z_calculated_half + 0.2, -0.08, r'$Z_{cal}/2$', fontsize=14, color='red')
        
        plt.text(0, 0.2, 'Accept H0', fontsize=16, color='green', ha='center')
        plt.text(-3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
        plt.text(3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
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

filename = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx" 
samples = load_sample_data(filename)
print(f"Total samples loaded: {len(samples)}")
sample1 = [player for player in samples if float(player[3]) < 20]  # Adjusted condition for Temperature
sample2 = [player for player in samples if float(player[3]) >= 20]  # Adjusted condition for Temperature
sample1 = sample1[:50] if len(sample1) > 50 else sample1
sample2 = sample2[:50] if len(sample2) > 50 else sample2
print(f"Sample1 size (Temp < 20°C): {len(sample1)}")
print(f"Sample2 size (Temp >= 20°C): {len(sample2)}")

if len(sample1) >= 30 and len(sample2) >= 30:
    tail = "two"  #one_left, one_right, two
    alpha = 0.05
    z_calculated, mean1, mean2, pooled_variance, var1, var2, critical_value = calculate_z(sample1, sample2, tail=tail, alpha=alpha)
    results = [{
        'Sample 1 Mean (Temp < 20°C)': mean1,
        'Sample 2 Mean (Temp >= 20°C)': mean2,
        'Pooled Variance': pooled_variance,
        'Variance1': var1,
        'Variance2': var2,
        'Calculated z': f"{z_calculated:.4f}",
        'Critical z': f"{critical_value:.4f}",
        'Result': "Reject H0" if (abs(z_calculated) > critical_value if tail == "two" else 
                                  (z_calculated < -critical_value if tail == "one_left" else z_calculated > critical_value)) else "Accept H0"
    }]
    print("\nZ-Test Case-3 Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    plot_hypothesis_test(z_calculated, critical_value, alpha=alpha, tail=tail)
else:
    print(f"Error: Sample sizes are n1 = {len(sample1)} and n2 = {len(sample2)}. Both must be at least 30 for z-test validity.")
