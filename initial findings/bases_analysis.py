import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Load the dataset
df = pd.read_excel('/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx')

# 2. Filter for HCT116 and prepare weights
cell_line = 'TE_HCT116'
df_filtered = df.dropna(subset=[cell_line, 'tx_sequence', 'utr5_size', 'utr3_size', 'cds_size']).copy()
df_filtered['weight'] = 2**df_filtered[cell_line]

# 3. Extract weighted nucleotide counts
utr5_counts = Counter()
utr3_counts = Counter()

for _, row in df_filtered.iterrows():
    seq = str(row['tx_sequence'])
    u5_len, u3_len, cds_len = int(row['utr5_size']), int(row['utr3_size']), int(row['cds_size'])
    weight = row['weight']
    
    # Slice sequences
    u5_seq = seq[:u5_len].upper()
    u3_start = u5_len + cds_len
    u3_seq = seq[u3_start : u3_start + u3_len].upper()
    
    for nt in u5_seq:
        if nt in 'ACGT': utr5_counts[nt] += weight
    for nt in u3_seq:
        if nt in 'ACGT': utr3_counts[nt] += weight

# 4. Format data for plotting
def counts_to_df(counts, region_name):
    total = sum(counts.values())
    return pd.DataFrame([{
        'Nucleotide': nt,
        'Percentage': (counts[nt] / total) * 100 if total > 0 else 0,
        'Region': region_name
    } for nt in ['A', 'C', 'G', 'T']])

combined_df = pd.concat([counts_to_df(utr5_counts, "5' UTR"), counts_to_df(utr3_counts, "3' UTR")])

# 5. Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=combined_df, x='Nucleotide', y='Percentage', hue='Region', palette='Set2')
plt.title(f'Nucleotide Composition in UTRs (Weighted by {cell_line})')
plt.ylabel('Percentage (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()