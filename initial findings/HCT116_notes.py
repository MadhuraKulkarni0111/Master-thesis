import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data 
# Note: Update the path to your actual file location
file_path = "/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# 2. Select the necessary columns
te_column = 'TE_HCT116' 
cols_to_keep = ['utr5_size', 'utr3_size', 'cds_size', te_column]
hct116_df = df[cols_to_keep].copy()

# Rename columns for easier access in the script
hct116_df.columns = ['utr5_length', 'utr3_length', 'cds_length', 'translation_efficiency']

# 3. Clean the data (remove rows with missing TE or length values)
hct116_df = hct116_df.dropna()

# 4. Create Length Bins (Short, Medium, Long)
hct116_df['utr5_bin'] = pd.qcut(hct116_df['utr5_length'], q=3, labels=['Short 5UTR', 'Medium 5UTR', 'Long 5UTR'])
hct116_df['utr3_bin'] = pd.qcut(hct116_df['utr3_length'], q=3, labels=['Short 3UTR', 'Medium 3UTR', 'Long 3UTR'])
hct116_df['cds_bin'] = pd.qcut(hct116_df['cds_length'], q=3, labels=['Short CDS', 'Medium CDS', 'Long CDS'])

# Calculate Non-coding region (5' + 3' UTR)
hct116_df['noncoding_length'] = hct116_df['utr5_length'] + hct116_df['utr3_length']
hct116_df['noncoding_bin'] = pd.qcut(hct116_df['noncoding_length'], q=3, labels=['Short NCR', 'Medium NCR', 'Long NCR'])

# 5. Plotting
def create_boxplot(column_bin, title, filename):
    plt.figure(figsize=(10, 6))
    hct116_df.boxplot(column='translation_efficiency', by=column_bin)
    plt.title(title)
    plt.suptitle("") # Remove default pandas title
    plt.xlabel("Length Category")
    plt.ylabel("Translation Efficiency")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Generate the plots
create_boxplot('utr5_bin', "TE vs 5' UTR Length", "te_vs_utr5.png")
create_boxplot('utr3_bin', "TE vs 3' UTR Length", "te_vs_utr3.png")
create_boxplot('cds_bin', "TE vs CDS Length", "te_vs_cds.png")
create_boxplot('noncoding_bin', "TE vs Non-coding Region Length", "te_vs_ncr.png")