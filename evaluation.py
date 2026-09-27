"""
Evaluation Module
Handles retraining with new data, model saving/loading, and predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def save_model(pipeline, filepath="models/best_model.pkl"):
    """Save the trained model pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"\n[OK] Model saved to: {filepath}")


def load_model(filepath="models/best_model.pkl"):
    """Load a trained model pipeline from disk."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Model file not found: {filepath}")
        return None
    try:
        pipeline = joblib.load(filepath)
        print(f"[OK] Model loaded from: {filepath}")
        return pipeline
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def retrain_model(pipeline, X_new, y_new, preprocessor=None):
    """
    Retrain the model with new data.
    Combines existing knowledge with new data.
    Returns the updated pipeline and training info.
    """
    print("\n" + "=" * 50)
    print("         MODEL RETRAINING")
    print("=" * 50)

    print(f"New data samples: {len(X_new)}")
    print(f"Features: {list(X_new.columns)}")

    # Fit on new data (pipeline retains preprocessing steps)
    pipeline.fit(X_new, y_new)
    print(f"\n[OK] Model retrained successfully")

    return pipeline


def retrain_with_combined(old_X_train, old_y_train, new_X, new_y, preprocessor, model_class, task_type):
    """
    Combine old and new data, then retrain.
    Returns updated pipeline and comparison info.
    """
    print("\n" + "=" * 50)
    print("         COMBINED RETRAINING")
    print("=" * 50)

    # Combine old and new data
    X_combined = pd.concat([old_X_train, new_X], ignore_index=True)
    y_combined = pd.concat([old_y_train, new_y], ignore_index=True)

    print(f"Old training samples: {len(old_X_train)}")
    print(f"New samples added:   {len(new_X)}")
    print(f"Combined total:      {len(X_combined)}")

    from .models import get_models

    models = get_models(task_type)
    results = {}

    for name, model_pipeline in models.items():
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_pipeline)
        ])

        full_pipeline.fit(X_combined, y_combined)

        # Cross-validation on combined data
        from sklearn.model_selection import cross_val_score
        if task_type == 'regression':
            scoring = 'r2'
        else:
            scoring = 'f1' if len(np.unique(y_combined)) == 2 else 'accuracy'

        cv_scores = cross_val_score(full_pipeline, X_combined, y_combined, cv=5, scoring=scoring)

        results[name] = {
            "pipeline": full_pipeline,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    # Find best
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_pipeline = results[best_name]["pipeline"]

    print(f"\n[OK] Best model after retraining: {best_name}")
    print(f"     CV Score: {results[best_name]['cv_mean']:.4f}")

    return best_pipeline, best_name, results


def predict_new_data(pipeline, feature_names):
    """
    Interactive prediction: ask user for feature values and predict.
    """
    print("\n" + "=" * 50)
    print("         PREDICTION")
    print("=" * 50)

    print(f"\nEnter feature values:")
    values = []
    for col in feature_names:
        while True:
            try:
                val = float(input(f"  {col}: "))
                values.append(val)
                break
            except ValueError:
                print(f"  [ERROR] Please enter a valid number")

    # Create DataFrame with proper column names
    X_new = pd.DataFrame([values], columns=feature_names)

    # Predict
    prediction = pipeline.predict(X_new)

    print(f"\nPrediction: {prediction[0]:.2f}")

    return prediction[0]


def run_experiments(df, target_col, task_type):
    """
    Run all experiments:
    1. Different train/test splits
    2. Model comparison
    3. Retraining with new data
    4. Prediction validation
    """
    from .preprocessing import preprocess_data
    from .training import train_models, select_best_model, evaluate_on_test
    from .models import get_models

    print("\n" + "=" * 50)
    print("         EXPERIMENTS")
    print("=" * 50)

    experiments = {}

    # Experiment 1: Different splits
    print("\n" + "-" * 40)
    print("EXPERIMENT 1: Train/Test Split Comparison")
    print("-" * 40)

    splits = [0.3, 0.2, 0.1]  # 70/30, 80/20, 90/10

    for test_size in splits:
        split_name = f"{int((1-test_size)*100)}/{int(test_size*100)}"
        print(f"\n--- Split: {split_name} ---")

        data = preprocess_data(df, target_col, task_type, test_size=test_size)
        if data is None:
            continue

        from sklearn.pipeline import Pipeline
        models = get_models(task_type)
        best_score = -float('inf')

        for name, model_pipeline in models.items():
            full_pipeline = Pipeline(steps=[
                ('preprocessor', data["preprocessor"]),
                ('model', model_pipeline)
            ])

            if task_type == 'regression':
                scoring = 'r2'
            else:
                scoring = 'f1' if len(np.unique(data["y_train"])) == 2 else 'accuracy'

            cv_scores = cross_val_score(full_pipeline, data["X_train"], data["y_train"], cv=5, scoring=scoring)
            mean_score = cv_scores.mean()

            if mean_score > best_score:
                best_score = mean_score

            print(f"  {name}: {mean_score:.4f}")

        experiments[f"split_{split_name}"] = best_score
        print(f"  Best: {best_score:.4f}")

    # Experiment 2: Model comparison on standard split
    print("\n" + "-" * 40)
    print("EXPERIMENT 2: Model Comparison (80/20 split)")
    print("-" * 40)

    data = preprocess_data(df, target_col, task_type, test_size=0.2)
    if data:
        results = train_models(
            data["preprocessor"], data["X_train"], data["y_train"], task_type
        )
        best_name, best_info = select_best_model(results, task_type)
        test_metrics = evaluate_on_test(best_info["pipeline"], data["X_test"], data["y_test"], task_type)
        experiments["model_comparison"] = {name: info["cv_mean"] for name, info in results.items()}
        experiments["test_metrics"] = {k: v for k, v in test_metrics.items() if k not in ["y_pred", "y_test"]}

    # Experiment 3: Retraining with new data
    print("\n" + "-" * 40)
    print("EXPERIMENT 3: Retraining with New Data")
    print("-" * 40)

    if data:
        # Simulate new data: take a subset as "new"
        split_idx = len(data["X_train"]) // 2
        X_old = data["X_train"].iloc[:split_idx]
        y_old = data["y_train"].iloc[:split_idx]
        X_new = data["X_train"].iloc[split_idx:]
        y_new = data["y_train"].iloc[split_idx:]

        print(f"\nOriginal training set: {len(X_old)} samples")
        print(f"New data added: {len(X_new)} samples")

        # Train on old data only
        models_old = get_models(task_type)
        old_results = {}
        for name, model_pipeline in models_old.items():
            full_pipeline = Pipeline(steps=[
                ('preprocessor', data["preprocessor"]),
                ('model', model_pipeline)
            ])
            if task_type == 'regression':
                scoring = 'r2'
            else:
                scoring = 'f1' if len(np.unique(y_old)) == 2 else 'accuracy'

            full_pipeline.fit(X_old, y_old)
            old_score = full_pipeline.score(data["X_test"], data["y_test"])
            old_results[name] = old_score
            print(f"  {name} (old data only): {old_score:.4f}")

        # Train on combined data
        X_combined = pd.concat([X_old, X_new], ignore_index=True)
        y_combined = pd.concat([y_old, y_new], ignore_index=True)

        models_new = get_models(task_type)
        new_results = {}
        for name, model_pipeline in models_new.items():
            full_pipeline = Pipeline(steps=[
                ('preprocessor', data["preprocessor"]),
                ('model', model_pipeline)
            ])
            full_pipeline.fit(X_combined, y_combined)
            new_score = full_pipeline.score(data["X_test"], data["y_test"])
            new_results[name] = new_score
            print(f"  {name} (combined data): {new_score:.4f}")

        # Show improvement
        print(f"\nImprovement after adding new data:")
        for name in old_results:
            if name in new_results:
                improvement = new_results[name] - old_results[name]
                print(f"  {name}: {old_results[name]:.4f} -> {new_results[name]:.4f} (delta: {improvement:+.4f})")

        experiments["retraining"] = {
            "old_results": old_results,
            "new_results": new_results
        }

    # Experiment 4: Prediction validation
    print("\n" + "-" * 40)
    print("EXPERIMENT 4: Prediction on Test Data")
    print("-" * 40)

    if data:
        best_pipeline = best_info["pipeline"]
        sample_idx = 0
        sample_X = data["X_test"].iloc[[sample_idx]]
        actual_y = data["y_test"].iloc[sample_idx]
        predicted_y = best_pipeline.predict(sample_X)[0]

        print(f"\nSample from test set:")
        print(f"  Features: {dict(sample_X.iloc[0])}")
        print(f"  Actual:    {actual_y}")
        print(f"  Predicted: {predicted_y:.2f}")
        print(f"  Error:     {abs(actual_y - predicted_y):.4f}")

        experiments["prediction_sample"] = {
            "actual": float(actual_y),
            "predicted": float(predicted_y),
            "error": float(abs(actual_y - predicted_y))
        }

    return experiments


def cross_val_score_import():
    """Import cross_val_score for use."""
    from sklearn.model_selection import cross_val_score
    return cross_val_score
