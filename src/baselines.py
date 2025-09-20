# src/baselines.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow

from src.utils import summarize_metrics

def run_baselines(X_train, y_train, X_test, y_test):
    """
    Trains, evaluates, and logs classical baseline models.
    """
    print("\n--- Running Baseline Classical Models ---")
    
    # We flatten the sequence data for these models.
    # Shape changes from (num_samples, seq_len, features) to (num_samples, seq_len * features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='rbf', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=name, # artifact_path is still used here, but we add an example
                input_example=X_train_flat[:5, :] # Provide a small sample of input data
            )
        # Start an MLflow run for each baseline model
        with mlflow.start_run(run_name=f"Baseline_{name}", nested=True):
            # Train the model
            model.fit(X_train_flat, y_train)
            
            # Make predictions on the test set
            y_pred = model.predict(X_test_flat)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {'accuracy': accuracy, 'f1_score': f1}
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            summarize_metrics(y_test, y_pred, title=f"{name} Test Metrics")
            
            # Log parameters and metrics to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_f1_score", f1)
            
            # Log the model to MLflow
            mlflow.sklearn.log_model(model, name)

    print("\n--- Baseline Model Comparison ---")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    return results_df