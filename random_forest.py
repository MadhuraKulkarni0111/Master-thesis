import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Helper function to calculate GC content
def calculate_gc(seq):
    if pd.isna(seq) or not isinstance(seq, str):
        return np.nan
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else np.nan

# 2. Main function for Random Forest Analysis
def run_random_forest_analysis(base_path, file_name, target_col, species_label):
    print(f"\n--- Starting Random Forest for {species_label} ({target_col}) ---")
    
    # Construct the full path
    full_path = os.path.join(base_path, file_name)
    
    # Load data (using read_excel for .xlsx files)
    df = pd.read_excel(full_path)
    
    # Feature Engineering
    if 'gc_content' not in df.columns:
        df['gc_content'] = df['tx_sequence'].apply(calculate_gc)
    
    features = ['tx_size', 'utr5_size', 'utr3_size', 'cds_size', 'gc_content']
    
    # Cleaning: Remove rows missing the target or features
    data = df[features + [target_col]].dropna()
    X = data[features]
    y = data[target_col]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train Random Forest
    # n_estimators=100: builds 100 decision trees
    # random_state=42: ensures reproducible results
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R-squared: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Feature Importance
    # Random Forest calculates importance based on how much each feature 
    # reduces variance (impurity) across all trees.
    importances = pd.DataFrame({
        'Feature': features, 
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importances:\n", importances)

    # Visualization: Feature Importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='magma')
    plt.title(f'Random Forest Feature Importance: {species_label}')
    plt.show()

# --- EXECUTION ---
file_path = '/Users/madhurakulkarni/Desktop/master_thesis/' 

# Run for Human
run_random_forest_analysis(file_path, 'Human_Data.xlsx', 'TE_HCT116', 'Human')

# Run for Mouse
run_random_forest_analysis(file_path, 'Mouse_Data.xlsx', 'TE_4T1', 'Mouse')