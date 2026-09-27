"""
Preprocessing Module
Handles automatic data preprocessing including missing values, encoding, and scaling.
Uses scikit-learn Pipeline and ColumnTransformer to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def identify_target(df, numeric_cols):
    """Try to auto-detect target column (last numeric column) or ask user."""
    if len(numeric_cols) < 2:
        print("[ERROR] Need at least 2 numeric columns (features + target)")
        return None

    # Heuristic: last numeric column is likely the target
    target_candidate = numeric_cols[-1]

    print(f"\nSuggested target column: '{target_candidate}'")
    print(f"Available numeric columns: {numeric_cols}")

    response = input(f"Use '{target_candidate}' as target? (y/n, or type column name): ").strip()

    if response.lower() == 'y' or response == '':
        return target_candidate
    elif response in df.columns:
        return response
    else:
        print("[ERROR] Invalid column name")
        return None


def detect_task_type(df, target_col):
    """Detect whether the task is regression or classification."""
    unique_values = df[target_col].nunique()

    # If column is object type, it's definitely classification
    if df[target_col].dtype == 'object':
        return 'classification'

    # If only 2 unique values (like 0/1), it's likely classification
    if unique_values <= 2:
        return 'classification'

    # If few unique values relative to dataset size, might be classification
    if unique_values <= 10 and unique_values / len(df) < 0.05:
        return 'classification'

    return 'regression'


def ask_task_type():
    """Ask user to confirm or override the task type."""
    print("\nTask type options:")
    print("  1. REGRESSION    — predict a continuous number")
    print("  2. CLASSIFICATION — predict a category (class)")

    response = input("Select task type (1 or 2): ").strip()
    if response == '1':
        return 'regression'
    elif response == '2':
        return 'classification'
    else:
        print("Invalid input, defaulting to REGRESSION")
        return 'regression'


def preprocess_data(df, target_col, task_type, test_size=0.2, random_state=42):
    """
    Preprocess the dataset:
    - Separate features and target
    - Split into train/test
    - Create preprocessing pipelines for numeric and categorical columns
    - Return preprocessed data and pipeline info
    """
    print("\n" + "=" * 50)
    print("         DATA PREPROCESSING")
    print("=" * 50)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numeric and categorical columns
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    print(f"\nFeatures ({len(X.columns)}):")
    print(f"  Numeric: {numeric_cols}")
    if categorical_cols:
        print(f"  Categorical: {categorical_cols}")
    print(f"  Target: {target_col}")

    # Check minimum data requirements
    if len(df) < 10:
        print("[ERROR] Too few rows for meaningful analysis (need at least 10)")
        return None

    # Build preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformers = [('num', numeric_transformer, numeric_cols)]

    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    print(f"Split ratio: {int((1-test_size)*100)}/{int(test_size*100)}")
    print(f"\n[OK] Preprocessing pipeline created")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
    }
