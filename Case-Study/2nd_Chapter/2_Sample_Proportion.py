import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate


# The null hypothesis assumes that there is no significant difference between the proportions of successes in the two groups. This is often stated as:

def two_sample_proportion_z_test(success_count1, success_count2, sample_size1, sample_size2, alpha=0.05, tail="two"):
    p_cap1 = success_count1 / sample_size1
    p_cap2 = success_count2 / sample_size2
    p_pool = (success_count1 + success_count2) / (sample_size1 + sample_size2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/sample_size1 + 1/sample_size2))
    z_calculated = (p_cap1 - p_cap2) / se if se > 0 else 0
    if tail == "two":
        z_tabulated = stats.norm.ppf(1 - alpha / 2)
    elif tail in ["one_left", "one_right"]:
        z_tabulated = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError(f"Invalid tail type '{tail}'. Use 'one_left', 'one_right', or 'two'.")  
    return z_calculated, z_tabulated, se, p_cap1, p_cap2, p_pool, (1 - p_pool), sample_size1, sample_size2

def get_results(success_count1, success_count2, sample_size1, sample_size2, alpha=0.05, tail="two"):
    try:
        z_calculated, z_tabulated, standard_error, p_cap1, p_cap2, p, q, n1, n2 = two_sample_proportion_z_test(
            success_count1, success_count2, sample_size1, sample_size2, alpha=alpha, tail=tail
        )
        if tail == "two":
            result = "Accept H0" if abs(z_calculated) <= z_tabulated else "Reject H0"
        elif tail == "one_left":
            result = "Accept H0" if z_calculated >= -z_tabulated else "Reject H0"
        elif tail == "one_right":
            result = "Accept H0" if z_calculated <= z_tabulated else "Reject H0"
    except ValueError as e:
        print(f"Error in two_sample_proportion_z_test: {e}")
        return None, None
    
    results = [{
        'Success Count (Group 1)': success_count1,
        'Success Count (Group 2)': success_count2,
        'Sample Size (n1)': n1,
        'Sample Size (n2)': n2,
        'Sample Proportion (p̂1)': f"{p_cap1:.4f}",
        'Sample Proportion (p̂2)': f"{p_cap2:.4f}",
        'Pooled Proportion (p)': p,
        'Q (1 - p)': f"{q:.4f}",
        'Alpha': alpha,
        'Tail': tail,
        'Z-calculated': f"{z_calculated:.4f}",
        'Z-tabulated': f"{z_tabulated:.4f}",
        'Result': result
    }]
    
    results_df = pd.DataFrame(results)
    print("\n2-Sample Proportion Test Results:")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

    return z_calculated, z_tabulated, standard_error

def plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail):
    x_values = np.linspace(-4, 4, 1000)
    y_values = stats.norm.pdf(x_values, 0, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Standard Normal Distribution', color='blue')
    if tail == "two":
        z_tabulated_half = z_tabulated / 2
        plt.fill_between(x_values, y_values, where=(x_values <= -z_tabulated_half), color='lightblue', alpha=0.5)
        plt.fill_between(x_values, y_values, where=(x_values >= z_tabulated_half), color='lightblue', alpha=0.5)
        plt.axvline(-z_tabulated_half, color='black', linestyle='--', label=f'-Z Tabulated/2 = {-z_tabulated_half:.2f}')
        plt.axvline(z_tabulated_half, color='black', linestyle='--', label=f'+Z Tabulated/2 = {z_tabulated_half:.2f}')  
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

# Load your dataset
file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Filter the data based on temperature condition
# For example, group1: Temperature > 15, group2: Temperature <= 15
group1 = df[df['Temperature'] > 15]
group2 = df[df['Temperature'] <= 15]

# Get success counts and sample sizes
success_count1 = len(group1)  # Count of days with temperature > 15
success_count2 = len(group2)  # Count of days with temperature <= 15
sample_size1 = len(df)  # Total sample size for group 1
sample_size2 = len(df)  # Total sample size for group 2

alpha = 0.05
tail = "two"  # one_left, one_right, two
z_calculated, z_tabulated, standard_error = get_results(success_count1, success_count2, sample_size1, sample_size2, alpha=alpha, tail=tail)
plot_hypothesis_test(z_calculated, z_tabulated, alpha, tail)
