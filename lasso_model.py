import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to compute GC content from sequence
def calculate_gc(seq):
    if pd.isna(seq) or not isinstance(seq, str):
        return np.nan
    return (seq.upper().count('G') + seq.upper().count('C')) / len(seq)

# Load your datasets
FILE_PATH = '/Users/madhurakulkarni/Desktop/master_thesis/'
human_df = pd.read_excel('Human_data.xlsx')
mouse_df = pd.read_excel('Mouse_Data.xlsx')

def run_lasso_analysis(df, target_col, title):
    # 1. Feature Engineering & Cleaning
    df['gc_content'] = df['tx_sequence'].apply(calculate_gc)
    features = ['tx_size', 'utr5_size', 'utr3_size', 'cds_size', 'gc_content']
    
    # Drop rows with missing values for the selected target or features
    clean_data = df.dropna(subset=[target_col] + features)
    
    X = clean_data[features]
    y = clean_data[target_col]
    
    # 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    model = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test)
    print(f"\n--- Model for {title} ---")
    print(f"R-squared: {r2_score(y_test, y_pred):.3f}")
    print("Coefficients:", dict(zip(features, model.coef_)))
    
    # 5. Visualization
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title(f'Lasso Model: Predicted vs Actual {title}')
    plt.xlabel(f'Experimental {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Run separate analyses
run_lasso_analysis(human_df, 'TE_HCT116', 'Human HCT116')
run_lasso_analysis(mouse_df, 'TE_4T1', 'Mouse 4T1')