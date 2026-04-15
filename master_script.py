import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from collections import Counter

# --- 1. CONFIGURATION & GENETIC CODE ---
FILE_PATH = '/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx'
CELL_LINE = 'TE_HCT116'  # Set your primary cell line for weighted analysis

GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

# --- 2. DATA LOADING & PREPROCESSING ---
def load_data(path):
    print(f"--- Loading data from {path} ---")
    df = pd.read_excel(path)
    # Filter rows missing essential sequence or TE data, imputation by dropping the values
    cols_to_check = [CELL_LINE, 'tx_sequence', 'utr5_size', 'utr3_size', 'cds_size']
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    # Linear weight calculation for biological frequency (from log2 TE)
    df_clean['weight'] = 2**df_clean[CELL_LINE]
    return df_clean

# --- 3. CORRELATION & REGRESSION ANALYSIS ---
def run_correlation_analysis(df):
    """Integrates spearman_correlation.py and plot.py logic."""
    print(f"\n--- Spearman Correlation Analysis for {CELL_LINE} ---")
    features = ['utr5_size', 'utr3_size', 'cds_size']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, feature in enumerate(features):
        coef, p_val = stats.spearmanr(df[feature], df[CELL_LINE])
        strength = "Strong" if abs(coef) > 0.4 else "Moderate" if abs(coef) > 0.2 else "Weak"
        print(f"{feature:10} | R: {coef:.4f} | P: {p_val:.2e} ({strength})")
        
        # Regression plot with log scale for sizes
        sns.regplot(data=df, x=feature, y=CELL_LINE, ax=axes[i],
                    scatter_kws={'alpha': 0.1, 's': 2}, line_kws={'color': 'red'})
        axes[i].set_xscale('log')
        axes[i].set_title(f"{feature} vs TE\nSpearman R: {coef:.2f}")
    
    plt.tight_layout()
    plt.show()

# --- 4. LENGTH BINNING ANALYSIS ---
def run_binning_analysis(df):
    """Integrates HCT116_notes.py boxplot logic."""
    print("\n--- Generating TE Distribution by Length Bins ---")
    df_bins = df.copy()
    
    # Calculate Non-coding region length
    df_bins['ncr_size'] = df_bins['utr5_size'] + df_bins['utr3_size']
    
    bin_targets = ['utr5_size', 'utr3_size', 'cds_size', 'ncr_size']
    labels = ['Short', 'Medium', 'Long']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(bin_targets):
        df_bins[f'{col}_bin'] = pd.qcut(df_bins[col], q=3, labels=labels)
        sns.boxplot(data=df_bins, x=f'{col}_bin', y=CELL_LINE, ax=axes[i], palette='Set3')
        axes[i].set_title(f'TE vs {col} Category')
        axes[i].set_ylabel('Translation Efficiency')
        
    plt.tight_layout()
    plt.show()

# --- 5. SEQUENCE COMPOSITION (AA, CODON, NT) ---
def run_sequence_composition(df):
    """Integrates amino_acid_analysis_hct116.py and bases_analysis.py."""
    print("\n--- Calculating Weighted Sequence Frequencies ---")
    codon_counts = Counter()
    aa_counts = Counter()
    utr5_nt, utr3_nt = Counter(), Counter()

    for _, row in df.iterrows():
        seq = str(row['tx_sequence']).upper()
        u5, u3, cds_len = int(row['utr5_size']), int(row['utr3_size']), int(row['cds_size'])
        w = row['weight']
        
        # CDS/AA Logic
        cds = seq[u5 : u5+cds_len]
        codons = [cds[i:i+3] for i in range(0, len(cds) - len(cds)%3, 3)]
        for c in codons:
            codon_counts[c] += w
            aa_counts[GENETIC_CODE.get(c, '?')] += w
            
        # UTR Nucleotide Logic
        u5_seq = seq[:u5]
        u3_seq = seq[u5 + cds_len : u5 + cds_len + u3]
        for nt in u5_seq: 
            if nt in 'ACGT': utr5_nt[nt] += w
        for nt in u3_seq: 
            if nt in 'ACGT': utr3_nt[nt] += w

    # Plot 1: Nucleotide Composition
    nt_data = []
    for region, counts in [("5' UTR", utr5_nt), ("3' UTR", utr3_nt)]:
        total = sum(counts.values())
        for nt in 'ACGT':
            nt_data.append({'Nucleotide': nt, 'Percentage': (counts[nt]/total)*100, 'Region': region})
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=pd.DataFrame(nt_data), x='Nucleotide', y='Percentage', hue='Region', palette='Set2')
    plt.title(f'Weighted Nucleotide Composition in UTRs ({CELL_LINE})')
    plt.show()

    # Plot 2: AA and Codon Frequencies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    aa_df = pd.DataFrame.from_dict(aa_counts, orient='index', columns=['count']).reset_index().rename(columns={'index':'AA'})
    aa_df['%'] = (aa_df['count'] / aa_df['count'].sum()) * 100
    sns.barplot(data=aa_df.sort_values('%', ascending=False), x='AA', y='%', ax=ax1, palette='viridis')
    ax1.set_title('Weighted Amino Acid Frequency (%)')

    codon_df = pd.DataFrame.from_dict(codon_counts, orient='index', columns=['count']).reset_index().rename(columns={'index':'Codon'})
    codon_df['%'] = (codon_df['count'] / codon_df['count'].sum()) * 100
    sns.barplot(data=codon_df.sort_values('%', ascending=False).head(25), x='Codon', y='%', ax=ax2, palette='magma')
    ax2.set_title('Top 25 Weighted Codons (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        data = load_data(FILE_PATH)
        
        # 1. Statistical Analysis (Correlations & Bins)
        run_correlation_analysis(data)
        run_binning_analysis(data)
        
        # 2. Molecular Composition Analysis
        run_sequence_composition(data)
        
        print("\n[SUCCESS] Master Analysis Routine Finished.")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Check file path and column names (TE_HCT116, tx_sequence, etc.)")