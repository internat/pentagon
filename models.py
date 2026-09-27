"""
Models Module
Defines mathematical models for regression and classification tasks.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


def get_regression_models():
    """Return a dictionary of regression models with their names."""
    models = {
        "Linear Regression": Pipeline(steps=[
            ('model', LinearRegression())
        ]),
        "Polynomial Regression (degree=2)": Pipeline(steps=[
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', LinearRegression())
        ]),
        "Decision Tree Regressor": Pipeline(steps=[
            ('model', DecisionTreeRegressor(
                max_depth=5,
                random_state=42
            ))
        ]),
        "Random Forest Regressor": Pipeline(steps=[
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ))
        ]),
    }
    return models


def get_classification_models():
    """Return a dictionary of classification models with their names."""
    models = {
        "Logistic Regression": Pipeline(steps=[
            ('model', LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Decision Tree Classifier": Pipeline(steps=[
            ('model', DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ))
        ]),
        "Random Forest Classifier": Pipeline(steps=[
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ))
        ]),
    }
    return models


def get_models(task_type):
    """Get models based on task type."""
    if task_type == 'regression':
        return get_regression_models()
    elif task_type == 'classification':
        return get_classification_models()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
