"""
Custom Linear Regression Implementation
Implements Linear Regression from scratch using NumPy.
This demonstrates understanding of the mathematical principles behind the algorithm.

Mathematical Model:
    y = X * w + b
    
where:
    X - feature matrix (n_samples x n_features)
    y - target vector (n_samples x 1)
    w - weight vector (n_features x 1)
    b - bias (scalar)

Method of Least Squares:
    Find w, b that minimize the sum of squared errors:
    L = sum((y_i - (X_i * w + b))^2)

Two approaches:
    1. Normal Equation (analytical solution):
       w = (X^T * X)^(-1) * X^T * y
       
    2. Gradient Descent (iterative optimization):
       w_new = w - lr * (dL/dw)
       where dL/dw = -2/n * X^T * (y - X*w)
"""

import numpy as np


class CustomLinearRegression:
    """
    Linear Regression from scratch.
    
    Supports two methods:
    - Normal Equation: closed-form analytical solution
    - Gradient Descent: iterative optimization
    """
    
    def __init__(self, method='normal_equation', learning_rate=0.01, n_iterations=1000):
        """
        Parameters:
            method: 'normal_equation' or 'gradient_descent'
            learning_rate: step size for gradient descent
            n_iterations: number of iterations for gradient descent
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model on data.
        
        Parameters:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X, y, n_samples, n_features)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y, n_samples, n_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fit_normal_equation(self, X, y, n_samples, n_features):
        """
        Normal Equation:
        w = (X^T * X)^(-1) * X^T * y
        
        This gives the exact optimal solution in one step.
        """
        # Add bias column (column of ones)
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # Normal equation: theta = (X^T * X)^(-1) * X^T * y
        try:
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_gradient_descent(self, X, y, n_samples, n_features):
        """
        Gradient Descent:
        Repeat until convergence:
            y_pred = X * w + b
            error = y_pred - y
            dw = (1/n) * X^T * error
            db = (1/n) * sum(error)
            w = w - lr * dw
            b = b - lr * db
        
        Note: Features are scaled internally for stable convergence.
        """
        # Scale features for gradient descent stability
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1  # avoid division by zero
        X_scaled = (X - self._mean) / self._std
        
        # Scale target too
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        if self._y_std == 0:
            self._y_std = 1
        y_scaled = (y - self._y_mean) / self._y_std
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X_scaled.dot(self.weights) + self.bias
            
            # Compute error
            error = y_pred - y_scaled
            
            # Compute gradients
            dw = (1 / n_samples) * X_scaled.T.dot(error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Track cost
            cost = np.mean(error ** 2)
            self.cost_history.append(cost)
        
        # Un-scale weights and bias back to original feature space
        # y_scaled = w_scaled * X_scaled + b_scaled
        # y = w_scaled * y_std * (X - mean)/std + b_scaled * y_std + y_mean
        # y = (w_scaled * y_std / std) * X + (b_scaled * y_std + y_mean - sum(w_scaled * y_std * mean / std))
        self.weights = self.weights * self._y_std / self._std
        self.bias = self.bias * self._y_std + self._y_mean - np.sum(self.weights * self._mean)
    
    def predict(self, X):
        """Make predictions."""
        X = np.array(X, dtype=float)
        return X.dot(self.weights) + self.bias
    
    def get_params(self):
        """Return model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'method': self.method,
        }


def compare_with_sklearn(X_train, y_train, X_test, y_test):
    """
    Compare custom implementation with sklearn's LinearRegression.
    """
    from sklearn.linear_model import LinearRegression
    
    print("\n" + "=" * 50)
    print("  COMPARISON: Custom vs Sklearn Linear Regression")
    print("=" * 50)
    
    # Custom model - Normal Equation
    custom_normal = CustomLinearRegression(method='normal_equation')
    custom_normal.fit(X_train, y_train)
    pred_normal = custom_normal.predict(X_test)
    
    # Custom model - Gradient Descent
    custom_gd = CustomLinearRegression(method='gradient_descent', learning_rate=0.1, n_iterations=1000)
    custom_gd.fit(X_train, y_train)
    pred_gd = custom_gd.predict(X_test)
    
    # Sklearn model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    pred_sklearn = sklearn_model.predict(X_test)
    
    # Compute metrics
    from sklearn.metrics import mean_squared_error, r2_score
    
    results = {
        'Custom (Normal Eq)': {
            'predictions': pred_normal,
            'RMSE': np.sqrt(mean_squared_error(y_test, pred_normal)),
            'R2': r2_score(y_test, pred_normal),
            'weights': custom_normal.weights,
            'bias': custom_normal.bias,
        },
        'Custom (Grad Desc)': {
            'predictions': pred_gd,
            'RMSE': np.sqrt(mean_squared_error(y_test, pred_gd)),
            'R2': r2_score(y_test, pred_gd),
            'weights': custom_gd.weights,
            'bias': custom_gd.bias,
        },
        'Sklearn': {
            'predictions': pred_sklearn,
            'RMSE': np.sqrt(mean_squared_error(y_test, pred_sklearn)),
            'R2': r2_score(y_test, pred_sklearn),
            'weights': sklearn_model.coef_,
            'bias': sklearn_model.intercept_,
        },
    }
    
    # Print comparison table
    print(f"\n{'Method':<25} {'RMSE':<12} {'R2':<12}")
    print("-" * 49)
    for name, info in results.items():
        print(f"{name:<25} {info['RMSE']:<12.4f} {info['R2']:<12.4f}")
    
    # Print parameters
    print(f"\nModel Parameters:")
    print(f"{'Method':<25} {'Weights':<20} {'Bias':<10}")
    print("-" * 55)
    for name, info in results.items():
        w_str = ", ".join([f"{v:.4f}" for v in info['weights']]) if info['weights'] is not None else "N/A"
        b_str = f"{info['bias']:.4f}" if info['bias'] is not None else "N/A"
        print(f"{name:<25} [{w_str}]     {b_str}")
    
    # Verify closeness
    print(f"\nVerification:")
    diff_normal_sklearn = np.max(np.abs(pred_normal - pred_sklearn))
    diff_gd_sklearn = np.max(np.abs(pred_gd - pred_sklearn))
    print(f"  Max difference (Normal Eq vs Sklearn): {diff_normal_sklearn:.6f}")
    print(f"  Max difference (Grad Desc vs Sklearn): {diff_gd_sklearn:.6f}")
    
    if diff_normal_sklearn < 1e-6:
        print("  [OK] Normal Equation matches Sklearn perfectly!")
    elif diff_normal_sklearn < 0.01:
        print("  [OK] Normal Equation is very close to Sklearn!")
    else:
        print("  [!] Normal Equation differs from Sklearn")
    
    if diff_gd_sklearn < 0.1:
        print("  [OK] Gradient Descent converges to similar solution!")
    else:
        print("  [!] Gradient Descent may need more iterations or tuning")
    
    return results
