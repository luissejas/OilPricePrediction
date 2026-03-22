import pandas as pd
from src.models.xgboost_model import XGBoostForecaster

def run_hyperparameter_search():
    print("--- Starting Hyperparameter Experimentation ---")
    
    # Define the different "Challengers" we want to test
    experiments = [
        {"name": "Fast & Shallow", "n_estimators": 85, "max_depth": 3, "lr": 0.1},
        {"name": "Fast & Shallow_more_estimators_v1", "n_estimators": 75, "max_depth": 3, "lr": 0.1},
        {"name": "Fast & Shallow_more_estimators_v2", "n_estimators": 80, "max_depth": 3, "lr": 0.1},
        {"name": "Fast & Shallow_more_estimators_v3", "n_estimators": 85, "max_depth": 3, "lr": 0.1},
        {"name": "Fast & Shallow_more_estimators_v4", "n_estimators": 90, "max_depth": 3, "lr": 0.1}
    ]
        # Champion 85 estimators
        # Second run shows more estimators worked better, will try one more rund
        # Previous models that did not work
        # {"name": "Standard", "n_estimators": 100, "max_depth": 5, "lr": 0.1},
        # {"name": "Deep & Slow", "n_estimators": 200, "max_depth": 7, "lr": 0.05},
        # {"name": "Robust Random", "n_estimators": 150, "max_depth": 4, "lr": 0.08}

    results = []

    for exp in experiments:
        # Initialize the model with the specific experiment parameters
        forecaster = XGBoostForecaster(
            n_estimators=exp["n_estimators"],
            max_depth=exp["max_depth"],
            learning_rate=exp["lr"]
        )
        
        forecaster.load_data()
        mae = forecaster.train_and_evaluate(model_name=exp["name"])
        
        results.append({"model": exp["name"], "mae": mae})

    # Sort results to find the "Champion" (lowest MAE)
    sorted_results = sorted(results, key=lambda x: x['mae'])
    
    print("\n--- Final Rankings ---")
    for i, res in enumerate(sorted_results):
        status = "CHAMPION" if i == 0 else f"Rank {i+1}"
        print(f"{status}: {res['model']} with MAE: ${res['mae']:.2f}")

if __name__ == "__main__":
    run_hyperparameter_search()
