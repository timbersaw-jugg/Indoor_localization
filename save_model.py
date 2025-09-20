# save_model.py
import bentoml
import torch
import joblib
import os
from src.models import IndoorLocalizer
from src.utils import load_config

# This script packages your k-fold models and scalers into one BentoML model.

if __name__ == "__main__":
    config = load_config()
    model_dir = config['paths']['model_dir']
    
    # Dynamically find how many folds were successfully trained
    num_folds = 0
    for item in os.listdir(model_dir):
        if item.startswith("model_fold_") and item.endswith(".pth"):
            num_folds += 1

    if num_folds == 0:
        raise FileNotFoundError("No trained models found in the 'models/' directory. Please run the training pipeline first.")

    # This should be the total number of unique location classes in your dataset
    NUM_CLASSES = 105 

    models = {}
    scalers = {}

    print(f"Packaging {num_folds} trained models...")
    for i in range(1, num_folds + 1):
        # Load the scaler for this fold
        scaler_path = os.path.join(model_dir, f'scaler_fold_{i}.joblib')
        scalers[f'scaler_fold_{i}'] = joblib.load(scaler_path)

        # Recreate model architecture and load its trained weights
        dummy_encoder = torch.nn.Sequential() # Placeholder for initialization
        model = IndoorLocalizer(config, NUM_CLASSES, pretrained_encoder=dummy_encoder)
        model_path = os.path.join(model_dir, f'model_fold_{i}.pth')
        model.load_state_dict(torch.load(model_path, map_location="cpu"),strict=False)
        model.eval()
        models[f'model_fold_{i}'] = model
    
    # Combine all models and scalers into a single dictionary to be saved
    saved_model_and_scalers = {"models": models, "scalers": scalers}

    # Save the combined dictionary as a single BentoML model
    bento_model = bentoml.pytorch.save_model(
        "indoor_localization_ensemble", # model name
        saved_model_and_scalers,        # the object to save
    )
    print(f"Bento model saved with tag: {bento_model.tag}")