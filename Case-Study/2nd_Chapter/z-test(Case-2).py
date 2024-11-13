import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate
# population varince known and variance  not equal

#There is no significant difference between the mean temperatures of group1 and group2.
def two_sample_z_test(mean1, mean2, std_dev1, std_dev2, size1, size2, alpha=0.05, tail="two"):
    standard_error = math.sqrt((std_dev1**2 / size1) + (std_dev2**2 / size2))
    z_calculated = (mean1 - mean2) / standard_error
    if tail == "two":
        z_tabulated = stats.norm.ppf(1 - alpha / 2)
    elif tail in ["one_left", "one_right"]:
        z_tabulated = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError(f"Invalid tail type '{tail}'. Use 'one_left', 'one_right', or 'two'.")
    return z_calculated, z_tabulated, standard_error

def get_results(mean1, mean2, std_dev1, std_dev2, size1, size2, alpha=0.05, tail="two"):
    try:
        z_calculated, z_tabulated, standard_error = two_sample_z_test(
            mean1, mean2, std_dev1, std_dev2, size1, size2, alpha=alpha, tail=tail
        )
        
        if tail == "two":
            result = "Accept H0" if abs(z_calculated) <= z_tabulated else "Reject H0"
        elif tail == "one_left":
            result = "Accept H0" if z_calculated >= -z_tabulated else "Reject H0"
        elif tail == "one_right":
            result = "Accept H0" if z_calculated <= z_tabulated else "Reject H0"
    except ValueError as e:
        print(f"Error in two_sample_z_test: {e}")
        return None, None
    
    results = [{
        'Mean 1': mean1,
        'Mean 2': mean2,
        'Std Dev 1': std_dev1,
        'Std Dev 2': std_dev2,
        'Size 1': size1,
        'Size 2': size2,
        'Alpha': alpha,
        'Tail': tail,
        'Z-calculated': f"{z_calculated:.4f}",
        'Z-tabulated': f"{z_tabulated:.4f}",
        'Result': result
    }]
    
    results_df = pd.DataFrame(results)
    print("\nZ-Test (Case-2) Results:")
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

file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Select data for groups, e.g., comparing Temperature in two categories
# You can adjust this as per your logic for group splitting
group1 = df[df['Temperature'] >= 15]['Temperature']  # Adjust the condition as needed
group2 = df[df['Temperature'] < 15]['Temperature']   # Adjust the condition as needed

# Ensure both groups have data before proceeding
if len(group1) == 0 or len(group2) == 0:
    print("One or both groups have no data. Please check the data and conditions.")
else:
    mean1 = group1.mean()
    mean2 = group2.mean()
    std_dev1 = group1.std(ddof=1)
    std_dev2 = group2.std(ddof=1)
    alpha = 0.05
    tail = "two"  # One of ["one_left", "one_right", "two"]
    
    # Run the Z-test and display results
    lower_bound, upper_bound = get_results(mean1, mean2, std_dev1, std_dev2, len(group1), len(group2), alpha=alpha, tail=tail)
