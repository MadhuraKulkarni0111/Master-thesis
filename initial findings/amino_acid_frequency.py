import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Load the dataset
# Replace 'your_file.csv' with the actual path to your file
df = pd.read_excel('/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx')

# 2. Define the Standard Genetic Code
genetic_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def extract_cds_and_counts(row):
    """Extracts CDS from tx_sequence and returns codon/AA counts."""
    seq = str(row['tx_sequence'])
    u5 = int(row['utr5_size'])
    cds_len = int(row['cds_size'])
    
    # Slice the sequence to get the Coding Region (CDS)
    cds_seq = seq[u5:u5+cds_len]
    
    # Break into codons (triplets)
    codons = [cds_seq[i:i+3] for i in range(0, len(cds_seq) - len(cds_seq)%3, 3)]
    
    # Translate to amino acids
    amino_acids = [genetic_code.get(c, '?') for c in codons]
    
    return Counter(codons), Counter(amino_acids)

# 3. Aggregate counts across the dataset (using a sample for speed)
total_codon_counts = Counter()
total_aa_counts = Counter()

for idx, row in df.iterrows():
    c_cnt, a_cnt = extract_cds_and_counts(row)
    total_codon_counts.update(c_cnt)
    total_aa_counts.update(a_cnt)

# 4. Convert to DataFrames for analysis
codon_df = pd.DataFrame.from_dict(total_codon_counts, orient='index', columns=['count']).reset_index().rename(columns={'index':'codon'})
codon_df['percentage'] = (codon_df['count'] / codon_df['count'].sum()) * 100
codon_df = codon_df.sort_values('percentage', ascending=False)

aa_df = pd.DataFrame.from_dict(total_aa_counts, orient='index', columns=['count']).reset_index().rename(columns={'index':'amino_acid'})
aa_df['percentage'] = (aa_df['count'] / aa_df['count'].sum()) * 100
aa_df = aa_df.sort_values('percentage', ascending=False)

# 5. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Amino Acid Plot
sns.barplot(data=aa_df, x='amino_acid', y='percentage', ax=ax1, palette='viridis')
ax1.set_title('Amino Acid Frequency Distribution (%)')
ax1.set_ylabel('Frequency (%)')

# Codon Plot (Top 30)
sns.barplot(data=codon_df.head(30), x='codon', y='percentage', ax=ax2, palette='magma')
ax2.set_title('Top 30 Codon Frequencies (%)')
ax2.set_ylabel('Frequency (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print summaries
print("Top 5 Amino Acids:\n", aa_df.head())
print("\nTop 5 Codons:\n", codon_df.head())