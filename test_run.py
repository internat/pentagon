"""
Non-interactive test script to verify all components work.
Runs the full pipeline without user input.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset, analyze_dataset, get_dataset_info
from src.preprocessing import preprocess_data, detect_task_type
from src.models import get_models
from src.training import train_models, select_best_model, evaluate_on_test
from src.evaluation import save_model, run_experiments
from src.visualization import plot_all_graphs
from src.custom_linear import compare_with_sklearn
from sklearn.model_selection import train_test_split


def test_regression():
    """Test the full regression pipeline."""
    print("\n" + "=" * 60)
    print("  TEST 1: REGRESSION PIPELINE")
    print("=" * 60)

    # Load data
    df = load_dataset("data/study_scores.csv")
    assert df is not None, "Failed to load dataset"

    # Analyze
    analyze_dataset(df)

    # Detect task
    task = detect_task_type(df, "test_score")
    assert task == "regression", f"Expected regression, got {task}"
    print(f"\nDetected task: {task}")

    # Preprocess
    data = preprocess_data(df, "test_score", "regression")
    assert data is not None, "Preprocessing failed"

    # Train
    results = train_models(
        data["preprocessor"], data["X_train"], data["y_train"], "regression"
    )
    assert len(results) > 0, "No models trained"

    # Compare
    best_name, best_info = select_best_model(results, "regression")

    # Evaluate
    test_metrics = evaluate_on_test(
        best_info["pipeline"], data["X_test"], data["y_test"], "regression"
    )
    assert "R2" in test_metrics, "R2 metric missing"
    print(f"\n[OK] Test R2 = {test_metrics['R2']:.4f}")

    # Save
    save_model(best_info["pipeline"])

    # Graphs
    feature_cols = list(data["X_train"].columns)
    plot_all_graphs(data, results, best_name, test_metrics, feature_cols[0], "test_score")

    # Custom Linear Regression demo
    X = df[["study_hours"]].values
    y = df["test_score"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    compare_with_sklearn(X_train, y_train, X_test, y_test)

    print("\n[OK] Regression pipeline PASSED")
    return test_metrics


def test_classification():
    """Test the full classification pipeline."""
    print("\n" + "=" * 60)
    print("  TEST 2: CLASSIFICATION PIPELINE")
    print("=" * 60)

    # Load data
    df = load_dataset("data/student_pass.csv")
    assert df is not None, "Failed to load dataset"

    # Analyze
    analyze_dataset(df)

    # Detect task
    task = detect_task_type(df, "pass")
    assert task == "classification", f"Expected classification, got {task}"
    print(f"\nDetected task: {task}")

    # Preprocess
    data = preprocess_data(df, "pass", "classification")
    assert data is not None, "Preprocessing failed"

    # Train
    results = train_models(
        data["preprocessor"], data["X_train"], data["y_train"], "classification"
    )
    assert len(results) > 0, "No models trained"

    # Compare
    best_name, best_info = select_best_model(results, "classification")

    # Evaluate
    test_metrics = evaluate_on_test(
        best_info["pipeline"], data["X_test"], data["y_test"], "classification"
    )
    assert "Accuracy" in test_metrics, "Accuracy metric missing"
    print(f"\n[OK] Test Accuracy = {test_metrics['Accuracy']:.4f}")

    print("\n[OK] Classification pipeline PASSED")
    return test_metrics


def test_experiments():
    """Test the experiments module."""
    print("\n" + "=" * 60)
    print("  TEST 3: EXPERIMENTS")
    print("=" * 60)

    df = load_dataset("data/study_scores.csv")
    experiments = run_experiments(df, "test_score", "regression")
    assert experiments is not None, "Experiments failed"

    print("\n[OK] Experiments PASSED")
    return experiments


def test_predictions():
    """Test prediction functionality."""
    print("\n" + "=" * 60)
    print("  TEST 4: PREDICTIONS")
    print("=" * 60)

    df = load_dataset("data/study_scores.csv")
    data = preprocess_data(df, "test_score", "regression")
    models = get_models("regression")

    from sklearn.pipeline import Pipeline

    # Get first model
    name = list(models.keys())[0]
    model = models[name]

    full_pipeline = Pipeline(steps=[
        ('preprocessor', data["preprocessor"]),
        ('model', model)
    ])
    full_pipeline.fit(data["X_train"], data["y_train"])

    # Predict
    test_vals = pd.DataFrame([[5.0]], columns=["study_hours"])
    pred = full_pipeline.predict(test_vals)
    print(f"\nPrediction for study_hours=5.0: {pred[0]:.2f}")
    print(f"Expected range: ~75-85")

    assert 70 < pred[0] < 90, f"Prediction {pred[0]} seems off"
    print("[OK] Predictions PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("  AUTOMATED TEST SUITE")
    print("=" * 60)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        test_regression()
        test_classification()
        test_experiments()
        test_predictions()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
