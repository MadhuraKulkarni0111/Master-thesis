import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Load the dataset
df = pd.read_excel('/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx')

# 2. Select Cell Line and Filter Data
cell_line = 'TE_HCT116'
df_filtered = df.dropna(subset=[cell_line, 'tx_sequence', 'utr5_size', 'cds_size']).copy()

# Convert log2(TE) to linear scale to use as weights for frequency calculation
df_filtered['weight'] = 2**df_filtered[cell_line]

# 3. Define Standard Genetic Code
genetic_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

# 4. Calculate Weighted Frequencies
codon_counts = Counter()
aa_counts = Counter()

for _, row in df_filtered.iterrows():
    seq, u5, c_len, weight = str(row['tx_sequence']), int(row['utr5_size']), int(row['cds_size']), row['weight']
    cds = seq[u5:u5+c_len]
    codons = [cds[i:i+3] for i in range(0, len(cds) - len(cds)%3, 3)]
    
    for c in codons:
        if len(c) == 3:
            codon_counts[c] += weight
            aa_counts[genetic_code.get(c, '?')] += weight

# Format results for plotting
aa_df = pd.DataFrame.from_dict(aa_counts, orient='index', columns=['count']).reset_index().rename(columns={'index':'AA'})
aa_df['percentage'] = (aa_df['count'] / aa_df['count'].sum()) * 100
aa_df = aa_df.sort_values('percentage', ascending=False)

codon_df = pd.DataFrame.from_dict(codon_counts, orient='index', columns=['count']).reset_index().rename(columns={'index':'Codon'})
codon_df['percentage'] = (codon_df['count'] / codon_df['count'].sum()) * 100
codon_df = codon_df.sort_values('percentage', ascending=False)

# 5. Generate Plots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Top: Amino Acid & Codon Frequencies
sns.barplot(data=aa_df, x='AA', y='percentage', ax=axs[0, 0], palette='viridis')
axs[0, 0].set_title(f'Weighted AA Frequency (%) - {cell_line}')

sns.barplot(data=codon_df.head(25), x='Codon', y='percentage', ax=axs[0, 1], palette='magma')
axs[0, 1].set_title(f'Top 25 Codons (%) - {cell_line}')
axs[0, 1].tick_params(axis='x', rotation=45)

# Bottom: Understanding Data Behavior
sns.scatterplot(data=df_filtered, x='utr5_size', y=cell_line, alpha=0.3, ax=axs[1, 0], color='teal')
axs[1, 0].set_title(f'5\' UTR Size vs {cell_line}')
axs[1, 0].set_xscale('log')

sns.scatterplot(data=df_filtered, x='cds_size', y=cell_line, alpha=0.3, ax=axs[1, 1], color='coral')
axs[1, 1].set_title(f'CDS Size vs {cell_line}')
axs[1, 1].set_xscale('log')

plt.tight_layout()
plt.show()