import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''with open('41587_2025_2712_MOESM3_ESM.xlsx') as f:  
   print(f.read()[0:200])'''

try:
    df = pd.read_excel("/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx", sheet_name=0) 
    # If the file name is the original .xlsx:
    # df = pd.read_excel("41587_2025_2712_MOESM3_ESM.xlsx", sheet_name=0) 
    
    print("File loaded successfully using read_excel().")
    print(df.head())

except Exception as e:
    print(f"Failed to load file as Excel: {e}")
    print("\nIf the file is named '...xlsx - Human.csv', try renaming it to remove the '.csv' suffix, or manually convert it to a *proper* CSV.")

# 1. Define the features (X) and TEs (Y) to compare
features = ['utr5_size', 'utr3_size', 'cds_size']
te_columns = ['TE_A549', 'TE_HeLa', 'TE_HCT116']

# Combine all columns of interest and remove rows with missing data
cols_of_interest = features + te_columns
df_clean = df[cols_of_interest].dropna()

# 2. Calculate the correlation matrix (Pearson R)
correlation_matrix = df_clean[cols_of_interest].corr(method='pearson')

# Extract and display the correlations between features and TEs
correlation_df = correlation_matrix.loc[features, te_columns]
print("--- Correlation (Pearson R) between Transcript Features and TE ---")
print(correlation_df)

# 3. Create a Visualization (Scatter plot for UTR3 size vs TE_A549)
# This visually demonstrates the correlation.

# Set up the plot aesthetics
plt.figure(figsize=(8, 6))
sns.regplot(
    x='utr3_size',
    y='TE_A549',
    data=df_clean,
    scatter_kws={'alpha': 0.1, 's': 5}, # Use smaller and transparent dots for large data
    line_kws={'color': 'red'},
    label=f"R: {correlation_df.loc['utr3_size', 'TE_A549']:.2f}"
)


# Add labels and title
plt.title(r'Translational Efficiency vs. 3\' UTR Size in A549 Cells', fontsize=14)
plt.xlabel(r'3\' UTR Size (Nucleotides)', fontsize=12)
plt.ylabel(r'Translational Efficiency ($\text{TE}_\text{A549}$)', fontsize=12)
plt.xscale('log') # UTR sizes and TE are typically analyzed on a log scale due to large ranges
plt.yscale('log')
plt.legend(title='Correlation', loc='upper left')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()

# Save the plot
plt.savefig('utr3_vs_te_a549_scatter.png')