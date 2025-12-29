import pandas as pd
import numpy as np
import os

def fix_test_data(input_path='data/test.csv', output_path='data/test_fixed.csv'):
    """Fix test data by removing label column if present"""
    print(f"Reading test data from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return None
    
    # Read the data
    test_data = pd.read_csv(input_path)
    print(f"Original shape: {test_data.shape}")
    print(f"Columns: {test_data.columns.tolist()}")
    
    # Check if first column looks like labels
    first_col = test_data.columns[0]
    if first_col.lower() in ['label', 'labels', 'target', 'y']:
        print(f"⚠️  First column '{first_col}' appears to be labels")
        print("Removing label column...")
        
        # Save features only
        features = test_data.iloc[:, 1:]
        features.to_csv(output_path, index=False)
        print(f"✅ Saved fixed test data to: {output_path}")
        print(f"New shape: {features.shape}")
        
        return output_path
    elif test_data.shape[1] == 785:
        print(f"⚠️  Data has 785 columns (expected 784 for MNIST)")
        
        # Check if first column contains numbers 0-9
        first_col_values = test_data.iloc[:, 0]
        unique_values = first_col_values.unique()
        if len(unique_values) <= 10 and all(0 <= val <= 9 for val in unique_values):
            print(f"First column contains digits: {sorted(unique_values)}")
            print("Removing label column...")
            
            features = test_data.iloc[:, 1:]
            features.to_csv(output_path, index=False)
            print(f"✅ Saved fixed test data to: {output_path}")
            print(f"New shape: {features.shape}")
            
            return output_path
        else:
            print("❓ First column doesn't look like digit labels")
            print(f"Unique values: {unique_values[:10]}...")
    
    print("✓ Test data appears to be correct (784 columns)")
    return input_path

def check_data_structure():
    """Check the structure of data files"""
    print("🔍 Checking data structure...")
    
    files_to_check = ['data/train.csv', 'data/test.csv']
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, nrows=5)
                print(f"\n📊 {file_path}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns (first 5): {df.columns.tolist()[:5]}")
                print(f"   First row sample: {df.iloc[0, :5].tolist()}")
                
                # Check for label column
                if df.shape[1] == 785:
                    print(f"   ⚠️  Has 785 columns (likely includes label)")
                    if df.columns[0].lower() in ['label', 'labels']:
                        print(f"   First column name: '{df.columns[0]}' (looks like label)")
                elif df.shape[1] == 784:
                    print(f"   ✓ Has 784 columns (features only)")
                else:
                    print(f"   ❓ Unexpected number of columns: {df.shape[1]}")
                    
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    print("=" * 50)
    print("DATA FIXING UTILITY")
    print("=" * 50)
    
    # Check data structure
    check_data_structure()
    
    # Fix test data if needed
    fixed_path = fix_test_data()
    
    if fixed_path:
        print(f"\n✅ Use this path in your code: '{fixed_path}'")
        print("\n📝 Update your main.py:")
        print(f'TEST_PATH = "{fixed_path}"')