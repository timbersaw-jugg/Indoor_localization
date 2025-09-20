# run.py

import os
import torch
import numpy as np
import pandas as pd
import mlflow
import argparse
import joblib
from torch.utils.data import DataLoader, TensorDataset

from src.utils import (load_config, summarize_metrics, plot_conf_mat, plot_error_cdf)
from src.data_processor import DataProcessor
from src.training import train_kfold_ensemble
from src.baselines import run_baselines
from src.evaluation import ensemble_predict

def main(args):
    """
    Main function to orchestrate the entire ML pipeline.
    """
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['plots_dir'], exist_ok=True)
    
    print("--- 1. Starting Data Processing ---")
    data_processor = DataProcessor(config)
    X_seq, y_seq, idx_to_coords = data_processor.load_and_prepare_data()
    X_train_val, X_test, y_train_val, y_test = data_processor.split_data(X_seq, y_seq)
    
    mlflow.set_experiment("Indoor Localization - Advanced Pipeline")
    
    with mlflow.start_run(run_name="Main_Pipeline_Run") as parent_run:
        print("\n--- 2. Started Main MLflow Run ---")
        mlflow.log_params(config['data'])

        trained_models = []
        if args.run_main_model or args.run_all:
            print("\n--- 3. Starting Main Model Training ---")
            trained_models = train_kfold_ensemble(config, X_train_val, y_train_val)
            
            print("\n--- 4. Starting Final Evaluation on Test Set ---")
            
            # Load the unique scaler saved for each fold
            scalers = []
            for fold in range(1, len(trained_models) + 1):
                scaler_path = os.path.join(config['paths']['model_dir'], f'scaler_fold_{fold}.joblib')
                scalers.append(joblib.load(scaler_path))
            
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_test).float()),
                batch_size=config['training']['batch_size']
            )
            
            final_preds = ensemble_predict(trained_models, scalers, test_loader, device)
            
            results_df = pd.DataFrame({'true_class': y_test, 'predicted_class': final_preds})
            results_df.to_csv('final_predictions.csv', index=False)
            print("Final predictions saved to `final_predictions.csv`")
            
            true_coords = np.array([idx_to_coords[c] for c in y_test])
            pred_coords = np.array([idx_to_coords[c] for c in final_preds])
            errors = np.linalg.norm(true_coords - pred_coords, axis=1)
            
            summarize_metrics(y_test, final_preds, title="Final Test Set Classification Report")
            plot_conf_mat(y_test, final_preds, config, title="Final Test Set Confusion Matrix")
            plot_error_cdf(errors, config)
            
            mean_error = np.mean(errors); median_error = np.median(errors)
            mlflow.log_metric("final_test_mean_error", mean_error)
            mlflow.log_metric("final_test_median_error", median_error)

        if args.run_baselines or args.run_all:
            print("\n--- 5. Starting Baseline Model Training ---")
            run_baselines(X_train_val, y_train_val, X_test, y_test)

    print("\n--- ✅ PIPELINE FINISHED SUCCESSFULLY ---")
    print("To view results, run: mlflow ui")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Indoor Localization ML Pipeline.")
    parser.add_argument('--run-main-model', action='store_true', help="Run only the main deep learning model training.")
    parser.add_argument('--run-baselines', action='store_true', help="Run only the baseline classical models.")
    parser.add_argument('--run-all', action='store_true', help="Run all parts of the pipeline (default).")
    args = parser.parse_args()

    if not any([args.run_main_model, args.run_baselines, args.run_all]):
        args.run_all = True
        
    main(args)