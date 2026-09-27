"""
Training Module
Handles model training with cross-validation and automatic best model selection.
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from .models import get_models


def train_models(preprocessor, X_train, y_train, task_type, cv_folds=5):
    """
    Train all models for the given task type.
    Returns a dictionary with model names, fitted models, and CV scores.
    """
    print("\n" + "=" * 50)
    print("         MODEL TRAINING")
    print("=" * 50)

    models = get_models(task_type)
    results = {}

    for name, model_pipeline in models.items():
        print(f"\n--- {name} ---")

        # Create full pipeline with preprocessor
        from sklearn.pipeline import Pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_pipeline)
        ])

        # Cross-validation scoring
        if task_type == 'regression':
            scoring = 'r2'
        else:
            scoring = 'f1' if len(np.unique(y_train)) == 2 else 'accuracy'

        cv_scores = cross_val_score(
            full_pipeline, X_train, y_train,
            cv=cv_folds, scoring=scoring
        )

        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        print(f"Cross-validation {scoring.upper()}:")
        print(f"  {mean_score:.4f} (+/- {std_score:.4f})")
        print(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}")

        # Fit on full training set
        full_pipeline.fit(X_train, y_train)

        results[name] = {
            "pipeline": full_pipeline,
            "cv_mean": mean_score,
            "cv_std": std_score,
            "cv_scores": cv_scores,
            "scoring": scoring,
        }

    return results


def select_best_model(results, task_type):
    """Select the best model based on CV scores."""
    print("\n" + "=" * 50)
    print("         MODEL COMPARISON")
    print("=" * 50)

    scoring = results[list(results.keys())[0]]["scoring"]

    # Sort by CV score (higher is better for both r2 and f1/accuracy)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["cv_mean"],
        reverse=True
    )

    print(f"\n{'Model':<35} {scoring.upper():<12}")
    print("-" * 48)
    for name, info in sorted_models:
        print(f"{name:<35} {info['cv_mean']:.4f} (+/- {info['cv_std']:.4f})")

    best_name = sorted_models[0][0]
    best_info = sorted_models[0][1]

    print(f"\n[OK] Best model: {best_name}")
    print(f"     CV Score: {best_info['cv_mean']:.4f} (+/- {best_info['cv_std']:.4f})")

    return best_name, best_info


def evaluate_on_test(best_pipeline, X_test, y_test, task_type):
    """Evaluate the best model on the test set."""
    print("\n" + "=" * 50)
    print("         FINAL MODEL EVALUATION (TEST SET)")
    print("=" * 50)

    y_pred = best_pipeline.predict(X_test)

    if task_type == 'regression':
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\nTest MAE:  {mae:.4f}")
        print(f"Test MSE:  {mse:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R2:   {r2:.4f}")

        print(f"\nMetric explanations:")
        print(f"  MAE  (Mean Absolute Error): Average absolute difference between")
        print(f"       predicted and actual values. Lower is better.")
        print(f"  MSE  (Mean Squared Error): Average squared difference. Penalizes")
        print(f"       large errors more. Lower is better.")
        print(f"  RMSE (Root Mean Squared Error): Square root of MSE. Same units")
        print(f"       as target variable. Lower is better.")
        print(f"  R2   (Coefficient of Determination): Proportion of variance in")
        print(f"       target explained by features. 1.0 = perfect, 0.0 = baseline.")

        return {
            "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "y_pred": y_pred, "y_test": y_test
        }

    else:  # classification
        accuracy = accuracy_score(y_test, y_pred)

        # Handle binary vs multiclass
        if len(np.unique(y_test)) == 2:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\nTest Accuracy:  {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-score:  {f1:.4f}")

        print(f"\nMetric explanations:")
        print(f"  Accuracy:  Proportion of correct predictions out of all predictions.")
        print(f"  Precision: Out of all predicted positives, how many are actually positive.")
        print(f"  Recall:    Out of all actual positives, how many were correctly identified.")
        print(f"  F1-score:  Harmonic mean of Precision and Recall. Balances both metrics.")

        return {
            "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1": f1,
            "y_pred": y_pred, "y_test": y_test
        }
