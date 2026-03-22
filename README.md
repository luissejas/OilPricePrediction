# Oil Price Prediction

This repository contains a machine learning pipeline for predicting daily crude oil closing prices (`CL=F`). It utilizes two distinct approaches—a PyTorch Neural Network and an XGBoost regressor—to generate and compare price forecasts.

## 📂 Repository Structure

* **`src/models/`**: Source code for model architectures and training logic.
* **`models/`**: Storage for trained model artifacts (`.pth`, `.json`) and the feature scaler (`.pkl`).
* **`interaction_GoogleColab/`**: Interactive notebook designed for cloud-based inference and visualization.
* **`data/`**: Directory for processed historical datasets.
* **`experiments/`**: Development scripts and notebooks used for testing and tuning.

---

## ⚡ Setup

Environment management is handled via [**uv**](https://github.com/astral-sh/uv).

### Local Environment
1.  Install `uv`.
2.  Run the following to install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

### Google Colab
The notebooks include a bootstrap cell that installs `uv` and synchronizes the environment automatically upon execution.

---

## 🛠️ Usage

### Training
Model training is performed by executing the scripts within `src/models/` or using the notebooks in the `experiments/` folder.

### Inference
The `interaction_GoogleColab/` directory contains the dashboard for running model inference. It fetches live data via the Yahoo Finance API and provides a slider interface to compare model predictions against actual Wall Street closing prices.

---
*Note: Market predictions are probabilistic and generated for research purposes.*
