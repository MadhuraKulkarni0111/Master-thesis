import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Helper function to calculate GC content from transcript sequences
def calculate_gc(seq):
    if pd.isna(seq) or not isinstance(seq, str):
        return np.nan
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else np.nan

# 2. Main function for Elastic Net Analysis
def run_elastic_net_analysis(file_path, target_col, species_label):
    print(f"\n--- Starting Analysis for {species_label} ({target_col}) ---")
    
    # Load data
    df = pd.read_excel(file_path)
    
    # Feature Engineering
    df['gc_content'] = df['tx_sequence'].apply(calculate_gc)
    features = ['tx_size', 'utr5_size', 'utr3_size', 'cds_size', 'gc_content']
    
    # Cleaning: Remove rows missing the target or features
    data = df[features + [target_col]].dropna()
    X = data[features]
    y = data[target_col]
    
    # Standardize Features (Crucial for Elastic Net)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # ElasticNetCV: Searches for best alpha (penalty) and l1_ratio (Lasso vs Ridge balance)
    # l1_ratio=1 is Lasso, l1_ratio=0 is Ridge.
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Optimal Alpha: {model.alpha_:.6f}")
    print(f"Optimal L1 Ratio: {model.l1_ratio_}")
    print(f"R-squared: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Coefficients Ranking
    coef_df = pd.DataFrame({'Feature': features, 'Weight': model.coef_}).sort_values(by='Weight', key=abs, ascending=False)
    print("Feature Weights:\n", coef_df)

    # Visualization 1: Regression Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f'Elastic Net: {species_label} Predicted vs Actual TE (R2={r2:.2f})')
    plt.xlabel('Experimental TE')
    plt.ylabel('Predicted TE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Visualization 2: Feature Importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Weight', y='Feature', data=coef_df, palette='viridis')
    plt.title(f'Elastic Net Feature Importance: {species_label}')
    plt.show()

# --- RUN ANALYSES ---
file_path = '/Users/madhurakulkarni/Desktop/master_thesis/' # ----- > change path 
run_elastic_net_analysis('Human_Data.xlsx', 'TE_HCT116', 'Human')
run_elastic_net_analysis('Mouse_Data.xlsx', 'TE_4T1', 'Mouse')