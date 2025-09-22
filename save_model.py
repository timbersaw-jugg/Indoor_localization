# save_model.py
import os
import torch
import joblib
import bentoml
from src.models import IndoorLocalizer, RSSIAutoencoder
from src.utils import load_config

BENTO_MODEL_NAME = "indoor_localization_ensemble"

if __name__ == "__main__":
    config = load_config()
    model_dir = config['paths']['model_dir']

    # --- Detect folds dynamically: only include folds with both model + scaler ---
    available_folds = []
    for i in range(1, 20):  # search up to 20 possible folds
        model_path = os.path.join(model_dir, f'model_fold_{i}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_fold_{i}.joblib')
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            available_folds.append(i)

    if len(available_folds) == 0:
        raise FileNotFoundError("No complete model+scaler pairs found. Please run the training pipeline first.")

    NUM_CLASSES = 105  # update if dataset changes

    models = {}
    scalers = {}

    print(f"Packaging {len(available_folds)} trained models...")

    for i in available_folds:
        # Load scaler
        scaler_path = os.path.join(model_dir, f'scaler_fold_{i}.joblib')
        scalers[f'scaler_fold_{i}'] = joblib.load(scaler_path)

        # Reconstruct encoder correctly
        temp_autoencoder = RSSIAutoencoder(config)
        correct_encoder = temp_autoencoder.encoder
        model = IndoorLocalizer(config, NUM_CLASSES, pretrained_encoder=correct_encoder)

        # Load trained weights
        model_path = os.path.join(model_dir, f'model_fold_{i}.pth')
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models[f'model_fold_{i}'] = model

    # --- Save as PicklableModel with custom_objects ---
    saved_objects = {"models": models, "scalers": scalers}

    bento_model = bentoml.picklable_model.save_model(
        name="indoor_localization_ensemble",
        model=saved_objects,
        custom_objects=saved_objects,
        
    )
    print(f"Bento model saved with tag: {bento_model.tag}")
