import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import openpyxl


# The null hypothesis is testing whether the average temperature or other measurements from the "before" group are equal to those from the "after" group, meaning there is no significant effect or change due to the intervention or condition you're studying.

def calculate_paired_t_test(x, y, alpha=0.05, tail="two"):
    differences = x - y
    d_bar = np.mean(differences)
    s_diff = np.std(differences, ddof=1)
    n = len(differences)
    s_d_bar = s_diff / np.sqrt(n)
    t_calculated = d_bar / s_d_bar
    df = n - 1
    if tail == "two":
        t_tabulated = stats.t.ppf(1 - alpha / 2, df)
    else:
        raise ValueError("Only two-tailed test is supported.")
    return t_calculated, t_tabulated

def load_sample_data(filename, num_samples=10):
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook.active
    temperature_data = []
    
    # Start reading data from the second row (index 2) to skip headers
    for row in sheet.iter_rows(min_row=2, values_only=True):
        try:
            temperature = float(row[3])  # Adjust index to match the Temperature column
            temperature_data.append(temperature)
        except (ValueError, TypeError):
            continue
    
    # Return the first num_samples entries and the next num_samples entries for before and after
    return np.array(temperature_data[:num_samples]), np.array(temperature_data[num_samples:num_samples * 2])

def plot_hypothesis_test(t_calculated, t_tabulated, alpha, tail):
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    t_calculated = abs(t_calculated)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue') 
    
    if tail == "two":
        t_tabulated_half = t_tabulated 
        plt.fill_between(x_values, y_values, where=(x_values <= -t_tabulated_half), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= t_tabulated_half), color='lightblue', alpha=0.5)
        plt.axvline(-t_tabulated_half, color='black', linestyle='--', label=f'-T Tabulated/2 = {-t_tabulated_half:.2f}')
        plt.axvline(t_tabulated_half, color='black', linestyle='--', label=f'+T Tabulated/2 = {t_tabulated_half:.2f}')  
        plt.axvline(-t_calculated, color='red', linestyle='-', label=f'-T Calculated = {-t_calculated:.2f}')
        plt.axvline(t_calculated, color='red', linestyle='-', label=f'+T Calculated = {t_calculated:.2f}')  
        
        plt.text(0, 0.2, 'Accept H0', fontsize=16, color='green', ha='center')
        plt.text(-3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
        plt.text(3.5, 0.02, 'Reject H0', fontsize=16, color='blue', ha='center')
    else:
        raise ValueError("Only two-tailed test is supported.")

    plt.title(f'{tail.capitalize()}-tailed T-Test: Hypothesis Testing Diagram')
    plt.xlabel('T value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

filename = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
before, after = load_sample_data(filename)

if len(before) > 0 and len(after) > 0:
    t_calculated, t_tabulated = calculate_paired_t_test(before, after)
    result = "Reject H0" if abs(t_calculated) > t_tabulated else "Accept H0"
    confidence_interval = np.mean(before - after) + np.array([-1, 1]) * t_tabulated * (np.std(before - after, ddof=1) / np.sqrt(len(before)))    
    
    print("Dependent T-test Results:")
    print("Samples taken:")
    print(f"Before: {before.tolist()}")
    print(f"After: {after.tolist()}")
    print(f"t-calculated: {t_calculated:.4f}")
    print(f"t-tabulated: {t_tabulated:.4f}")
    print(f"Result: {result}")
    print(f"Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    
    plot_hypothesis_test(t_calculated, t_tabulated, alpha=0.05, tail="two")
else:
    print("Error: One or both of the samples are empty.")
