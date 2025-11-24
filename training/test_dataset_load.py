"""
Test script to verify dataset loading from Kaggle
"""

import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("sakhawat18/asteroid-dataset")
print("Path to dataset files:", path)

# List files in the directory
print("\nFiles in dataset directory:")
for file in os.listdir(path):
    print(f"  - {file}")

# Try to load the CSV file
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    csv_path = os.path.join(path, csv_files[0])
    print(f"\nLoading CSV file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.info())
else:
    print("No CSV files found!")
