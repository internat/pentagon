"""
Data Loader and Analyzer
Handles loading CSV files and performing initial dataset analysis.
"""

import pandas as pd
import numpy as np


def load_dataset(filepath):
    """Load a CSV dataset and return a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Dataset loaded: {filepath}")
        print(f"     Rows: {len(df)}, Columns: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"[ERROR] File is empty: {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"[ERROR] File is corrupted or not a valid CSV: {filepath}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error loading file: {e}")
        return None


def analyze_dataset(df):
    """Perform automatic analysis of the dataset and display results."""
    print("\n" + "=" * 50)
    print("         DATASET ANALYSIS")
    print("=" * 50)

    # Basic info
    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn names: {list(df.columns)}")

    # Data types
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\nMissing values:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\nMissing values: None")

    # Unique values
    print(f"\nUnique values per column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count}")

    # Separate numeric and categorical columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    print(f"\nNumeric columns ({len(numeric_cols)}):")
    for col in numeric_cols:
        print(f"  {col}")

    if categorical_cols:
        print(f"\nCategorical columns ({len(categorical_cols)}):")
        for col in categorical_cols:
            print(f"  {col}")

    # Statistical description for numeric
    if numeric_cols:
        print(f"\nStatistical summary (numeric):")
        desc = df[numeric_cols].describe()
        print(desc.round(2).to_string())

    # Categorical summary
    if categorical_cols:
        print(f"\nCategorical summary:")
        for col in categorical_cols:
            print(f"\n  {col} value counts:")
            print(f"  {df[col].value_counts().to_dict()}")

    print("\n" + "=" * 50)
    return numeric_cols, categorical_cols


def get_dataset_info(df):
    """Return a dictionary with dataset information for internal use."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
