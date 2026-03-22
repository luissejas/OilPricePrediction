import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class XGBoostForecaster:
    """
    Trains an XGBoost regression model to predict crude oil prices.
    Accepts hyperparameters during initialization for experimentation.
    """
    def __init__(self, data_path="data/processed_oil_data.csv", n_estimators=100, learning_rate=0.1, max_depth=5):
        self.data_path = data_path
        # The model dynamically accepts variables for hyperparameter tuning
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        # Drop rows with missing values caused by rolling averages
        self.df.dropna(inplace=True)

    def train_and_evaluate(self, model_name="Default Model"):
        # Features (X) and Target (y)
        feature_cols = ['price_lag_1', 'price_lag_3', 'sma_7', 'sma_14', 'volatility_7']
        X = self.df[feature_cols]
        y = self.df['price']

        # Strict chronological split (80% train, 20% test) to prevent data leakage
        split_idx = int(len(self.df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        
        print(f"[{model_name}] Mean Absolute Error: ${mae:.2f} per barrel")
        return mae
    
    def save_model(self, file_path="models/champion_xgboost.json"):
        """Saves the trained model weights to a JSON file."""
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save_model(file_path)
        print(f"Model successfully saved to {file_path}")

    def load_model(self, file_path="models/champion_xgboost.json"):
        """Loads a pre-trained model from a JSON file."""
        self.model.load_model(file_path)
        print(f"Model successfully loaded from {file_path}")

if __name__ == "__main__":
    # Testing Version 1: Fast and Shallow
    model_v1 = XGBoostForecaster(n_estimators=85, learning_rate=0.1, max_depth=3)
    model_v1.load_data()
    model_v1.train_and_evaluate(model_name="V1 (Shallow)")
