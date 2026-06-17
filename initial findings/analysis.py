import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. LOAD AND PREPARE DATA
file_path = '/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM4_ESM.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Columns of interest
target_cell_line = 'TE_4T1' #FOR HUMAN: HCT116, MOUSE: TE_4T1
comparison_cell_line = 'TE_3T3' # For comparison plot, HUMAN: TE_HEK293, MOUSE: TE_3T3
cols = ['SYMBOL', 'utr5_size', 'utr3_size', 'cds_size', 'tx_sequence', target_cell_line, comparison_cell_line]

# Create a cleaned dataframe (removing rows where HCT116 TE is missing)
hct_df = df[cols].dropna(subset=[target_cell_line]).copy()

# 2. FEATURE ENGINEERING
# Function to calculate GC Content (%) from the sequence
def get_gc_content(seq):
    if pd.isna(seq): return np.nan
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq) * 100

print("Calculating GC content...")
hct_df['gc_content'] = hct_df['tx_sequence'].apply(get_gc_content)

# Total Non-coding Region (NCR) size
hct_df['ncr_size'] = hct_df['utr5_size'] + hct_df['utr3_size']

# Create Categories (Short, Medium, Long) using Quantiles
labels = ['Short', 'Medium', 'Long']
hct_df['utr5_bin'] = pd.qcut(hct_df['utr5_size'], q=3, labels=labels)
hct_df['utr3_bin'] = pd.qcut(hct_df['utr3_size'], q=3, labels=labels)
hct_df['cds_bin'] = pd.qcut(hct_df['cds_size'], q=3, labels=labels)
hct_df['ncr_bin'] = pd.qcut(hct_df['ncr_size'], q=3, labels=labels)

# 3. GENERATING THE PLOTS

# --- PLOT 1: Multi-panel Box Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
bins = ['utr5_bin', 'utr3_bin', 'cds_bin', 'ncr_bin']
titles = ["5' UTR Length", "3' UTR Length", "CDS Length", "Non-coding Region Length"]

for ax, b, t in zip(axes.flatten(), bins, titles):
    sns.boxplot(x=b, y=target_cell_line, data=hct_df, ax=ax, palette='Set2', showfliers=False)
    ax.set_title(f'TE vs {t}')
    ax.set_ylabel('Translation Efficiency')
    ax.set_xlabel('Category')

plt.suptitle('TE_4T1: Impact of Transcript Region Lengths on TE', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('1_TE_4T1_boxplots.png')

# --- PLOT 2: Scatter Plot with Regression (Log Scale) ---
plt.figure(figsize=(10, 6))
# Using log10 for length because biological lengths vary by orders of magnitude
sns.regplot(x=np.log10(hct_df['utr5_size'] + 1), y=target_cell_line, data=hct_df,
            scatter_kws={'alpha': 0.1, 's': 5}, line_kws={'color': 'red'})
plt.title("TE_4T1: Continuous Trend of 5' UTR Length vs TE")
plt.xlabel("Log10(5' UTR Length)")
plt.ylabel("Translation Efficiency (TE)")
plt.savefig('2_TE_4T1_scatter_regression.png')

# --- PLOT 3: Cell Line Comparison (HCT116 vs HEK293) ---
plt.figure(figsize=(8, 8))
plt.scatter(hct_df[comparison_cell_line], hct_df[target_cell_line], alpha=0.2, s=5, color='teal')
# Draw a diagonal line (y=x)
max_val = max(hct_df[target_cell_line].max(), hct_df[comparison_cell_line].max())
plt.plot([0, max_val], [0, max_val], 'r--', label='Equal Efficiency')
plt.xlabel(f'TE in {comparison_cell_line}')
plt.ylabel(f'TE in {target_cell_line}')
plt.title(f'Comparison: TE_4T1 vs {comparison_cell_line}')
plt.legend()
plt.savefig('3_TE_4T1_comparison.png')

# --- PLOT 4: GC Content Hexbin (Density) ---
plt.figure(figsize=(10, 8))
plt.hexbin(hct_df['gc_content'], hct_df[target_cell_line], gridsize=40, cmap='YlGnBu', mincnt=1)
plt.colorbar(label='Number of Genes')
plt.xlabel('GC Content (%)')
plt.ylabel('Translation Efficiency (TE)')
plt.title('TE_4T1: TE vs Transcript GC Content')
plt.savefig('4_TE_4T1_gc_density.png')

# --- PLOT 5: Violin Plot (Density Shape) ---
plt.figure(figsize=(10, 6))
sns.violinplot(x='utr5_bin', y=target_cell_line, data=hct_df, palette='Pastel1', inner='quartile')
plt.title("TE_4T1: Distribution Density of TE by 5' UTR Category")
plt.ylabel('Translation Efficiency')
plt.xlabel("5' UTR Length Category")
plt.savefig('5_TE_4T1_violin.png')

print("Success! All plots have been saved as PNG files.")