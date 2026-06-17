import csv
import sys
import matplotlib as plt
import pandas as pd

# *** 1. The Critical Fixes ***
# Use the full, correct file name of the CSV
file_path = "/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx"
# The encoding that fixed the UnicodeDecodeError
file_encoding = 'latin-1'
# The delimiter that fixed the ParserError
file_delimiter = ','

try:
    # Use standard Python 'open' to handle the file path and encoding
    with open(file_path, 'r', encoding=file_encoding) as csvfile:
        
        # Use the built-in 'csv.reader' to handle the delimiter
        # It reads each line and returns a list of values (fields)
        reader = csv.reader(csvfile, delimiter=file_delimiter)
        
        # --- Read and Print Header ---
        header = next(reader)
        print("--- HEADER ---")
        print(header)
        print(f"\nTotal Columns Found: {len(header)}\n")
        
        # --- Read and Print First 5 Data Rows ---
        print("--- FIRST 5 DATA ROWS ---")
        count = 0
        data_rows = []
        for row in reader:
            # Check the number of fields in each row for consistency
            if len(row) != len(header):
                print(f"!!! WARNING: Line {count+2} has {len(row)} fields, expected {len(header)}.")
                
            data_rows.append(row)
            print(row[:8], '...') # Print only the first 8 columns for brevity
            count += 1
            if count >= 5:
                break
                
        print("\nFile read successfully using built-in CSV module.")
        
except FileNotFoundError:
    print(f"\nERROR: File not found at the specified path: {file_path}")
    print("Please ensure the file name is exactly correct and is in the same directory as your script.")
except Exception as e:
    print(f"\nAn unexpected error occurred during file reading: {e}")

# If the read is successful, you can convert the data_rows into a list of lists 
# and then easily load it into a pandas DataFrame (if you need pandas later):
'''if 'data_rows' in locals():
    # If successful, you can put it into pandas now without file reading errors:
    df_resolved = pd.DataFrame(data_rows, columns=header)
        # Set up the plot aesthetics
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=df_clean,
        x='utr3_size',
        y='TE_A549',
        scatter_kws={'s': 10, 'alpha': 0.5},
        line_kws={'color': 'red'},
    )


    # Add labels and title
    plt.title(r'Translational Efficiency vs. 3\' UTR Size in A549 Cells', fontsize=14)
    plt.xlabel(r'3\' UTR Size (Nucleotides)', fontsize=12)
    plt.ylabel(r'Translational Efficiency ($\text{TE}_\text{A549}$)', fontsize=12)
    plt.xscale('log') # UTR sizes and TE are typically analyzed on a log scale due to large ranges
    plt.yscale('log')
    plt.legend(title='Correlation', loc='upper left')
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Save the plot
    plt.savefig('utr3_vs_te_a549_scatter.png')'''