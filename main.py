"""
Intelligent Data Analysis System
Main entry point with CLI menu.

A simple AutoML-style pipeline that:
- Loads data
- Analyzes structure
- Preprocesses automatically
- Detects task type
- Trains multiple models
- Compares and selects the best one
- Evaluates on test data
- Makes predictions
"""

import os
import sys
import pandas as pd
import numpy as np

from src.data_loader import load_dataset, analyze_dataset, get_dataset_info
from src.preprocessing import identify_target, detect_task_type, ask_task_type, preprocess_data
from src.models import get_models
from src.training import train_models, select_best_model, evaluate_on_test
from src.evaluation import (
    save_model, load_model, predict_new_data,
    run_experiments
)
from src.visualization import plot_all_graphs
from src.custom_linear import compare_with_sklearn


# Global state
class AppState:
    def __init__(self):
        self.df = None
        self.dataset_path = None
        self.target_col = None
        self.task_type = None
        self.data = None
        self.results = None
        self.best_name = None
        self.best_info = None
        self.test_metrics = None
        self.experiments = None


def print_banner():
    print()
    print("=" * 50)
    print("  INTELLIGENT DATA ANALYSIS SYSTEM")
    print("  AutoML Pipeline for Regression & Classification")
    print("=" * 50)


def print_menu():
    print()
    print("Menu:")
    print("  1. Load & Analyze dataset")
    print("  2. Train models")
    print("  3. Compare & Select best model")
    print("  4. Evaluate best model on test set")
    print("  5. Make prediction on new data")
    print("  6. Run all experiments")
    print("  7. Show all graphs")
    print("  8. Custom Linear Regression demo")
    print("  9. Save / Load model")
    print("  0. Exit")
    print()


def menu_load_analyze(state):
    """Load and analyze a dataset."""
    filepath = input("\nEnter CSV file path (e.g., data/study_scores.csv): ").strip()

    df = load_dataset(filepath)
    if df is None:
        return

    state.df = df
    state.dataset_path = filepath

    # Analyze
    analyze_dataset(df)

    # Auto-detect target
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    target = identify_target(df, numeric_cols)
    if target is None:
        return

    state.target_col = target

    # Auto-detect or ask task type
    task = detect_task_type(df, target)
    print(f"\nDetected task type: {task.upper()}")
    override = input("Override? (y/n): ").strip().lower()
    if override == 'y':
        task = ask_task_type()

    state.task_type = task
    print(f"\n[OK] Dataset ready: {len(df)} rows, target='{target}', task={task}")


def menu_train(state):
    """Train models on the dataset."""
    if state.df is None:
        print("\n[ERROR] Please load a dataset first (option 1)")
        return

    # Preprocess
    data = preprocess_data(state.df, state.target_col, state.task_type)
    if data is None:
        return

    state.data = data

    # Train
    results = train_models(
        data["preprocessor"], data["X_train"], data["y_train"], state.task_type
    )
    state.results = results

    print("\n[OK] All models trained successfully")


def menu_compare(state):
    """Compare models and select the best one."""
    if state.results is None:
        print("\n[ERROR] Please train models first (option 2)")
        return

    best_name, best_info = select_best_model(state.results, state.task_type)
    state.best_name = best_name
    state.best_info = best_info


def menu_evaluate(state):
    """Evaluate the best model on test data."""
    if state.best_info is None:
        print("\n[ERROR] Please select best model first (option 3)")
        return

    test_metrics = evaluate_on_test(
        state.best_info["pipeline"], state.data["X_test"], state.data["y_test"], state.task_type
    )
    state.test_metrics = test_metrics


def menu_predict(state):
    """Make a prediction on new data."""
    if state.best_info is None:
        print("\n[ERROR] Please train and select a model first (options 2-3)")
        return

    feature_names = list(state.data["X_train"].columns)
    predict_new_data(state.best_info["pipeline"], feature_names)


def menu_experiments(state):
    """Run all experiments."""
    if state.df is None:
        print("\n[ERROR] Please load a dataset first (option 1)")
        return

    experiments = run_experiments(state.df, state.target_col, state.task_type)
    state.experiments = experiments


def menu_graphs(state):
    """Generate all graphs."""
    if state.data is None or state.results is None or state.test_metrics is None:
        print("\n[ERROR] Please train models and evaluate first (options 2-4)")
        return

    # Find the first feature column for scatter plot
    feature_cols = list(state.data["X_train"].columns)
    feature_col = feature_cols[0]

    plot_all_graphs(
        state.data, state.results, state.best_name,
        state.test_metrics, feature_col, state.target_col
    )


def menu_custom_demo(state):
    """Run custom Linear Regression comparison."""
    if state.df is None:
        print("\n[ERROR] Please load a dataset first (option 1)")
        return

    from sklearn.model_selection import train_test_split

    X = state.df.drop(columns=[state.target_col]).values
    y = state.df[state.target_col].values

    # Use only first feature for simplicity
    if X.shape[1] > 1:
        print("\nNote: Using only the first feature for custom demo")
        X = X[:, 0].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    compare_with_sklearn(X_train, y_train, X_test, y_test)


def menu_save_load(state):
    """Save or load model."""
    print("\n  1. Save current model")
    print("  2. Load saved model")
    choice = input("Choice: ").strip()

    if choice == '1':
        if state.best_info is None:
            print("[ERROR] No model to save. Train first.")
            return
        save_model(state.best_info["pipeline"])

    elif choice == '2':
        pipeline = load_model()
        if pipeline:
            print("[OK] Model loaded. You can now make predictions.")
            # Store in state for use
            if state.best_info is None:
                state.best_info = {"pipeline": pipeline}
            else:
                state.best_info["pipeline"] = pipeline


def main():
    """Main program loop."""
    print_banner()
    state = AppState()

    while True:
        print_menu()
        choice = input("Select option: ").strip()

        if choice == '1':
            menu_load_analyze(state)
        elif choice == '2':
            menu_train(state)
        elif choice == '3':
            menu_compare(state)
        elif choice == '4':
            menu_evaluate(state)
        elif choice == '5':
            menu_predict(state)
        elif choice == '6':
            menu_experiments(state)
        elif choice == '7':
            menu_graphs(state)
        elif choice == '8':
            menu_custom_demo(state)
        elif choice == '9':
            menu_save_load(state)
        elif choice == '0':
            print("\nGoodbye!")
            break
        else:
            print("\n[ERROR] Invalid option. Please select 0-9.")


if __name__ == "__main__":
    main()
