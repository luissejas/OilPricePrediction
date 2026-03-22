# Oil Price Forecasting Engine
This project implements a dual-model machine learning pipeline to predict crude oil prices (CL=F). It integrates daily market data with specialized architectures in PyTorch and XGBoost to provide a comprehensive analysis of price trends.

## 📂 Repository Architecture
The project is organized into modular directories to maintain a clear distinction between code, data, and interactive tools.

## 1. Model Blueprints (src/models/)
Contains the structural definitions of the forecasting algorithms.

pytorch_model.py: The architectural class for the Deep Learning Multi-Layer Perceptron.

xgboost_model.py: The training and evaluation logic for the Gradient Boosted Tree.

## 2. Trained Artifacts (models/)
Stores the finalized "brains" and preprocessing tools.

champion_nn.pth: Optimized weights for the Neural Network.

champion_xgboost.json: The saved state of the trained XGBoost model.

nn_scaler.pkl: The standard scaler used to normalize input features.

## 3. Interactive Dashboard (interaction_GoogleColab/)
OilPricePrediction_Interaction.ipynb: A cloud-optimized notebook featuring a custom UI. It allows for real-time inference using a trading-day selection slider and live market data.

## 4. Supporting Infrastructure
data/: Secured storage for the processed_oil_data.csv baseline.

config/: Tracking files for performance metrics and champion scores.

tests/: Unit tests for verifying data loading and model initialization.

experiments/: A sandbox for feature engineering and hyperparameter tuning.

## 🚀 Key Functionalities
Side-by-Side Inference
The system runs two distinct algorithms simultaneously to provide a cross-verified prediction:

Deep Learning (PyTorch): Captures complex non-linear relationships in the price history.

Gradient Boosting (XGBoost): Provides robust, tree-based forecasting based on tabular technical indicators.

Daily Market Integration
The engine fetches the most recent daily closing prices via the Yahoo Finance API. This ensures that predictions are grounded in the most recent completed trading sessions rather than volatile intra-day fluctuations.

Cross-Device Hardware Support
The pipeline automatically detects the available hardware (CUDA GPU or standard CPU) and handles the weight mapping dynamically to ensure zero-error execution in any environment.

## 🛠️ Getting Started
Installation
Bash
git clone https://github.com/luissejas/OilPricePrediction.git
cd OilPricePrediction
pip install -r requirements.txt
Running the Dashboard
Upload the interaction_GoogleColab/ notebook to Google Colab.

Run the initialization cell to sync the repository.

Use the Trading Day Slider to select a date and evaluate the AI's performance against actual market closing prices.

## Disclaimer: This project is for research and educational purposes. Market predictions are probabilistic and do not constitute financial advice.
