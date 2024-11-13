import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate
import scipy.stats as stats
# population varince unknown and variance not equal
# Function to generate random sample data


# The null hypothesis (H0) in your z-test setup is that there is no significant difference in the mean temperatures between the two sample groups (sample1 with temperature < 450 and sample2 with temperature >= 450).

def generate_random_data(n_samples):
    data = []
    for i in range(n_samples):
        location = f"Location_{i}"
        temperature = np.random.uniform(100, 500)  # Random temperature between 100 and 500
        co2_emissions = np.random.uniform(0, 10)
        sea_level_rise = np.random.uniform(0, 3)
        precipitation = np.random.uniform(0, 10)
        humidity = np.random.uniform(30, 80)
        wind_speed = np.random.uniform(0, 20)
        data.append((location, "Country_X", temperature, co2_emissions, sea_level_rise, precipitation, humidity, wind_speed))
    return data

# Function to calculate the z-test statistics
def calculate_z(sample1, sample2):
    temp1 = [float(player[2]) for player in sample1]  # Updated index for Temperature column
    temp2 = [float(player[2]) for player in sample2]  # Updated index for Temperature column
    n1 = len(temp1)
    n2 = len(temp2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Samples must contain data.")
    mean1 = np.mean(temp1)
    mean2 = np.mean(temp2)
    var1 = np.var(temp1, ddof=1)  # Sample variance
    var2 = np.var(temp2, ddof=1)  # Sample variance
    z_calculated = abs(mean1 - mean2) / np.sqrt((var1 / (n1-1)) + (var2 / (n2-1)))
    return z_calculated, mean1, mean2, var1, var2

# Function to plot the hypothesis test
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
    plt.xlabel("z-value")
    plt.ylabel("Probability Density")
    plt.title("Hypothesis Test: Acceptance and Rejection Regions")
    plt.legend()
    plt.show()

# Generate random sample data
n_samples = 1000  # Total number of samples
samples = generate_random_data(n_samples)

# Separate the samples into two groups
sample1 = [player for player in samples if player[2] < 450]  # Temperature < 450
sample2 = [player for player in samples if player[2] >= 450]  # Temperature >= 450
sample1 = sample1[:50] if len(sample1) > 50 else sample1
sample2 = sample2[:50] if len(sample2) > 50 else sample2

# Check sample sizes
print(f"Sample1 size (Temperature < 450): {len(sample1)}")
print(f"Sample2 size (Temperature >= 450): {len(sample2)}")

# Perform the z-test if sample sizes are sufficient
if len(sample1) >= 30 and len(sample2) >= 30:
    z_calculated, mean1, mean2, var1, var2 = calculate_z(sample1, sample2)
    alpha = 0.05
    critical_value = norm.ppf(1 - alpha / 2)
    tail = "two"  # "one_left", "one_right", or "two"
    
    # Display results
    results = [{
        'Sample 1 Mean (Temperature < 450)': mean1,
        'Sample 2 Mean (Temperature >= 450)': mean2,
        'Variance1': var1,
        'Variance2': var2,
        'Calculated z': f"{z_calculated:.4f}",
        'Critical z': f"{critical_value:.4f}",
        'Result': "Reject H0" if abs(z_calculated) > critical_value else "Accept H0"
    }]
    print("\nZ-Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Plot the hypothesis test
    plot_hypothesis_test(z_calculated, critical_value, alpha=alpha, tail=tail)
else:
    print(f"Error: Sample sizes are n1 = {len(sample1)} and n2 = {len(sample2)}. Both must be at least 30 for z-test validity.")
