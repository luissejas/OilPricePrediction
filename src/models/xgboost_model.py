import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

class XGBoostForecaster:
    def __init__(self, data_path="data/processed_oil_data.csv", n_estimators=100, learning_rate=0.1, max_depth=5):
        self.data_path = data_path
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        self.df.dropna(inplace=True)

    def train_and_evaluate(self, model_name="Default Model"):
        feature_cols = ['price_lag_1', 'price_lag_3', 'sma_7', 'sma_14', 'volatility_7']
        X = self.df[feature_cols]
        y = self.df['price']

        split_idx = int(len(self.df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"[{model_name}] Mean Absolute Error: ${mae:.2f}")
        return mae

    # --- THE MISSING PIECE ---
    def save_model(self, filename="models/champion_xgboost.json"):
        # Ensure the root-level models folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.model.save_model(filename)
        print(f"✅ Model artifact saved to: {filename}")

if __name__ == "__main__":
    # 1. Initialize and Load
    forecaster = XGBoostForecaster(n_estimators=85, learning_rate=0.1, max_depth=3)
    forecaster.load_data()
    
    # 2. Train
    forecaster.train_and_evaluate(model_name="V1 (Shallow)")
    
    # 3. Save to the root 'models/' folder
    forecaster.save_model()
