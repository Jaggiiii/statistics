import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Load the data from the provided file
file_path = "/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
df = pd.read_excel(file_path)

# Check if the DataFrame is not empty
if not df.empty:
    # Filter the rows based on the Temperature column (adjusted for this dataset)
    filtered_df = df[df['Temperature'] > 10]  # Modify this filter as per your needs
    if len(filtered_df) >= 4:
        # Randomly select 4 values from the filtered "Temperature" column
        random_values = filtered_df['Temperature'].sample(n=4, random_state=None).values
        a, b, c, d = random_values
        print(f"\nValues of a, b, c, d:")
        print(f"{a:<20} {b:<20}")
        print(f"{c:<20} {d:<20}\n")
        
        # Adjust the condition to ensure no values are below a threshold
        threshold = 10  # You can modify this as needed
        if any(value <= threshold for value in [a, b, c, d]):
            print(f"Error: One or more values are less than or equal to {threshold}")
        else:
            # Calculate Chi-Square Yates statistic (for demonstration purposes)
            N = a + b + c + d
            chi2_yates = (N * (a * d - b * c) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d))
            dof = 1  # Degrees of freedom
            p_value = 1 - chi2.cdf(chi2_yates, dof)
            alpha = 0.05  # Significance level
            chi_tab_value = chi2.ppf(1 - alpha, dof)  # Tabulated value for comparison
            result = "Reject H0" if p_value < alpha else "Accept H0"
            
            # Display results
            print("\nChi-Square Test Results:")
            print(f"Chi-square Statistic: {chi2_yates:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Degrees of Freedom: {dof}")
            print(f"Tabulated Value: {chi_tab_value:.4f}")
            print(f"Result: {result}")
            
            # Plot the Chi-Square Distribution and the test statistic
            x = np.linspace(0, chi2_yates + 10, 1000)
            y = chi2.pdf(x, dof)
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label='Chi-Square Distribution', color='blue')
            plt.fill_between(x, 0, y, where=(x >= chi2_yates), color='red', alpha=0.5, label='Right Tail')
            plt.axvline(chi2_yates, color='green', linestyle='--', label=f'Chi-Square Statistic ({chi2_yates:.2f})')
            plt.axvline(chi_tab_value, color='orange', linestyle='--', label=f'Tabulated Value ({chi_tab_value:.2f})')
            plt.text(chi2_yates + 0.5, max(y) * 0.8, f'a: {a:.2f}\nb: {b:.2f}\nc: {c:.2f}\nd: {d:.2f}', 
                     fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(chi2_yates + 0.5, max(y) * 0.6, f'Result: {result}', 
                     fontsize=12, color='black', bbox=dict(facecolor='yellow', alpha=0.5))
            plt.xlabel('Chi-Square Value')
            plt.ylabel('Probability Density')
            plt.title("Right-Tailed Test Visualization for Chi-Square Distribution")
            plt.legend()
            plt.show()
    else:
        print(f"Error: Not enough values greater than threshold in the 'Temperature' column for sampling.")
else:
    print("Error: The DataFrame is empty.")
