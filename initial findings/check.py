import pandas as pd
import sys

file_path = "/Users/madhurakulkarni/Desktop/master_thesis/41587_2025_2712_MOESM3_ESM.xlsx - Human.csv"
encodings = ['utf-8', 'latin-1', 'cp1252']
separators = [',', ';', '\t'] # Comma, Semicolon, Tab

found = False

print(f"--- Attempting to read '{file_path}' with different encodings and delimiters ---")

for encoding in encodings:
    for sep in separators:
        try:
            # We only read 5 rows to make the test quick
            df_test = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=5)
            
            # Check if the number of columns is reasonable (we expect > 5 based on the headers)
            if df_test.shape[1] > 5:
                print(f"\nSUCCESS: Read file using encoding='{encoding}' and sep='{sep}'")
                print("Columns appear separated correctly.")
                print("\nDataFrame Head (first 5 rows):")
                print(df_test.head())
                found = True
                break # Stop inner loop
            
        except UnicodeDecodeError:
            # If a UnicodeDecodeError occurs, we just try the next encoding
            pass 
        except pd.errors.ParserError:
            # If a ParserError occurs, the separator is likely wrong, so we try the next separator
            pass 
        except Exception as e:
            # Catch other potential errors
            pass 

    if found:
        break # Stop outer loop

if not found:
    print("\nFAILURE: Could not successfully read the file with the common encoding/separator combinations.")
    print("If you see the file head above, use that combination. Otherwise, please visually inspect the file.")