"""
Visualization Module
Creates graphs for data analysis, model comparison, and predictions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os


def ensure_results_dir():
    """Ensure results directory exists."""
    os.makedirs("results", exist_ok=True)


def plot_data_and_model(X_train, y_train, X_test, y_test, pipeline, feature_col, target_col):
    """
    Graph 1: Scatter plot of data with model prediction line.
    For single-feature regression only.
    """
    ensure_results_dir()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training data
    ax.scatter(X_train[feature_col], y_train, color='blue', alpha=0.6,
               label='Training data', edgecolors='black', linewidth=0.5)

    # Plot test data
    ax.scatter(X_test[feature_col], y_test, color='red', alpha=0.6,
               label='Test data', edgecolors='black', linewidth=0.5)

    # Create prediction line
    x_range = np.linspace(X_train[feature_col].min(), X_train[feature_col].max(), 200).reshape(-1, 1)
    import pandas as pd
    x_range_df = pd.DataFrame(x_range, columns=[feature_col])

    # Add other feature columns as zeros (for pipeline compatibility)
    for col in X_train.columns:
        if col != feature_col:
            x_range_df[col] = 0

    y_pred_line = pipeline.predict(x_range_df)
    ax.plot(x_range.flatten(), y_pred_line, color='green', linewidth=2,
            label='Model prediction')

    ax.set_xlabel(feature_col, fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.set_title(f'{target_col} vs {feature_col} — Model Fit', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/01_data_and_model.png", dpi=150)
    plt.close()
    print("[OK] Graph 1 saved: results/01_data_and_model.png")


def plot_actual_vs_predicted(y_test, y_pred, task_type):
    """
    Graph 2: Actual vs Predicted scatter plot.
    """
    ensure_results_dir()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))

    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5, color='steelblue')

    # Perfect prediction line
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect prediction (y = x)')

    ax.set_xlabel('Actual values', fontsize=12)
    ax.set_ylabel('Predicted values', fontsize=12)
    ax.set_title('Actual vs Predicted', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig("results/02_actual_vs_predicted.png", dpi=150)
    plt.close()
    print("[OK] Graph 2 saved: results/02_actual_vs_predicted.png")


def plot_model_comparison(model_names, cv_scores, task_type):
    """
    Graph 3: Bar chart comparing model quality.
    """
    ensure_results_dir()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ['steelblue' if s < max(cv_scores) else 'forestgreen' for s in cv_scores]

    bars = ax.barh(model_names, cv_scores, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10)

    scoring_name = "R2" if task_type == "regression" else "F1"
    ax.set_xlabel(f'{scoring_name} Score', fontsize=12)
    ax.set_title('Model Comparison — Cross-Validation Scores', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/03_model_comparison.png", dpi=150)
    plt.close()
    print("[OK] Graph 3 saved: results/03_model_comparison.png")


def plot_error_distribution(y_test, y_pred):
    """
    Graph 4: Error distribution histogram.
    """
    ensure_results_dir()

    errors = np.array(y_test) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error histogram
    axes[0].hist(errors, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Error over samples
    axes[1].bar(range(len(errors)), errors, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Error', fontsize=12)
    axes[1].set_title('Error per Sample', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/04_error_distribution.png", dpi=150)
    plt.close()
    print("[OK] Graph 4 saved: results/04_error_distribution.png")


def plot_all_graphs(data, results, best_name, test_metrics, feature_col, target_col):
    """Generate all graphs."""
    print("\n" + "=" * 50)
    print("         GENERATING GRAPHS")
    print("=" * 50)

    try:
        # Graph 1: Data + model
        plot_data_and_model(
            data["X_train"], data["y_train"],
            data["X_test"], data["y_test"],
            results[best_name]["pipeline"],
            feature_col, target_col
        )

        # Graph 2: Actual vs Predicted
        plot_actual_vs_predicted(
            test_metrics["y_test"], test_metrics["y_pred"],
            "regression" if "R2" in test_metrics else "classification"
        )

        # Graph 3: Model comparison
        model_names = list(results.keys())
        cv_scores = [results[name]["cv_mean"] for name in model_names]
        task_type = "regression" if "R2" in test_metrics else "classification"
        plot_model_comparison(model_names, cv_scores, task_type)

        # Graph 4: Error distribution (regression only)
        if "R2" in test_metrics:
            plot_error_distribution(test_metrics["y_test"], test_metrics["y_pred"])

        print(f"\n[OK] All graphs saved to results/")

    except Exception as e:
        print(f"[ERROR] Error generating graphs: {e}")
        import traceback
        traceback.print_exc()
