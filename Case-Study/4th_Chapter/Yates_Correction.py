import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'
df = pd.read_excel(file_path)

if not df.empty:
    # Let's filter the data based on a specific condition, e.g., Temperature > 10
    filtered_df = df[df['Temperature'] > 10]

    if len(filtered_df) >= 4:
        # Sample random values from the Temperature column
        random_values = filtered_df['Temperature'].sample(n=4, random_state=None).values
        a, b, c, d = random_values

        print("\nValues from Random Sampling:")
        print(f"{a:<20} {b:<20}")
        print(f"{c:<20} {d:<20}\n")

        # Calculate N (total count) and Yates' corrected Chi-Square statistic
        N = a + b + c + d
        chi2_yates = (N * ((abs(a * d - b * c) - (N / 2)) ** 2) /
                      ((a + b) * (c + d) * (a + c) * (b + d)))

        dof = 1  # Degrees of freedom for a 2x2 contingency table
        alpha = 0.05
        chi_tab_value = chi2.ppf(1 - alpha, dof)  # Critical value
        p_value = 1 - chi2.cdf(chi2_yates, dof)
        result = "Reject H0" if p_value < alpha else "Accept H0"

        print("\nChi-Square Test Results with Yates Correction:")
        print(f"Chi-square Statistic: {chi2_yates:.4f}")
        print(f"Chi-Tab Value (Critical Value at alpha = {alpha}): {chi_tab_value:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of Freedom: {dof}")
        print(f"Result: {result}")

        # Plotting the Chi-Square distribution and test result
        x = np.linspace(0, chi2_yates + 10, 1000)
        y = chi2.pdf(x, dof)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Chi-Square Distribution', color='blue')
        plt.fill_between(x, 0, y, where=(x >= chi2_yates), color='red', alpha=0.5, label='Right Tail')
        plt.axvline(chi2_yates, color='green', linestyle='--', label=f'Chi-Square Statistic ({chi2_yates:.2f})')
        plt.axvline(chi_tab_value, color='purple', linestyle='--', label=f'Chi-Tab Value ({chi_tab_value:.2f})')

        # Adding labels and title
        plt.xlabel('Chi-Square Value')
        plt.ylabel('Probability Density')
        plt.title("Right-Tailed Test Visualization for Chi-Square Distribution")
        plt.legend()
        plt.show()

    else:
        print("Error: Not enough values greater than 10 in the 'Temperature' column for sampling.")
else:  
    print("Error: The DataFrame is empty.")
