import pandas as pd
import scipy.stats as stats

# 1. Load data
df = pd.read_excel('/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx')
cell_line = 'TE_HCT116'

# 2. Clean data (ensure we have sizes and TE values)
cols_to_check = [cell_line, 'utr5_size', 'cds_size', 'utr3_size']
df_clean = df.dropna(subset=cols_to_check).copy()

# 3. Calculate Spearman Correlation
# We use Spearman because the relationship is often non-linear
features = ['utr5_size', 'cds_size', 'utr3_size']

print(f"--- Spearman Correlation Analysis for {cell_line} ---")
for feature in features:
    coef, p_value = stats.spearmanr(df_clean[feature], df_clean[cell_line])
    
    # Interpretation
    strength = "Strong" if abs(coef) > 0.4 else "Moderate" if abs(coef) > 0.2 else "Weak"
    direction = "Negative" if coef < 0 else "Positive"
    
    print(f"{feature:10} | Correlation: {coef:.4f} | P-value: {p_value:.2e} ({strength} {direction})")

# 4. Quick Comparison (Mean Sizes of High vs Low TE genes)
high_te = df_clean[df_clean[cell_line] > df_clean[cell_line].median()]
low_te = df_clean[df_clean[cell_line] <= df_clean[cell_line].median()]

print("\n--- Average Sizes by TE Group ---")
print(f"High TE Genes Avg CDS: {high_te['cds_size'].mean():.1f} bp")
print(f"Low TE Genes Avg CDS:  {low_te['cds_size'].mean():.1f} bp")