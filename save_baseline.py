import os
import pandas as pd
from utils.data_processing import load_all_data, DEFAULT_DATA_DIR
from utils.models import train_models

def save_baseline_results():
    print("Loading data...")
    final_data, _, _ = load_all_data(DEFAULT_DATA_DIR)
    
    print("Training baseline models...")
    model_state = train_models(final_data.to_json(orient="split"))
    
    # Create results directory
    os.makedirs("results/baseline", exist_ok=True)
    
    # Save metrics
    metrics = model_state["metrics"]
    metrics.to_csv("results/baseline/baseline_metrics.csv", index=False)
    
    # Save feature importance
    importance = model_state["importance"]
    importance.to_csv("results/baseline/baseline_feature_importance.csv", index=False)
    
    # Save predictions for test set (to calculate ROC/AUC later if needed)
    predictions = model_state["predictions"]
    predictions.to_csv("results/baseline/baseline_predictions.csv", index=False)
    
    print("Baseline results saved successfully in 'results/baseline/' directory.")

if __name__ == "__main__":
    save_baseline_results()
