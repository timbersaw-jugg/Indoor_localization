# service.py

import bentoml
import numpy as np
import torch
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "indoor_localization_ensemble:latest"

# Load the Bento model we will create in the next step
# The custom_objects are our saved models and scalers
runner = bentoml.pytorch.get(BENTO_MODEL_TAG).to_runner()

# Create the API service
svc = bentoml.Service("indoor_localization_classifier", runners=[runner])

@svc.api(input=NumpyNdarray(shape=(-1, 20, 13), dtype=np.float32), output=NumpyNdarray())
def predict(input_sequence: np.ndarray) -> np.ndarray:
    """
    This is the API function that will be called to make predictions.
    """
    all_logits = []
    device = torch.device("cpu") # Forcing CPU for simplicity in deployment
    
    # Access the models and scalers from the loaded Bento
    models_dict = runner.custom_objects['models']
    scalers_dict = runner.custom_objects['scalers']

    # Perform ensemble prediction
    for i in range(1, len(models_dict) + 1):
        model = models_dict[f'model_fold_{i}']
        scaler = scalers_dict[f'scaler_fold_{i}']
        model.to(device)
        
        # Scale the input data using the correct scaler for this fold's model
        nsamples, nsteps, nfeatures = input_sequence.shape
        input_reshaped = input_sequence.reshape(-1, nfeatures)
        input_scaled = scaler.transform(input_reshaped).reshape(nsamples, nsteps, nfeatures)
        input_tensor = torch.from_numpy(input_scaled).float().to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            all_logits.append(outputs.cpu())
            
    # Average the logits across all models (soft voting)
    avg_logits = torch.stack(all_logits).mean(dim=0)
    final_preds = avg_logits.argmax(dim=1)
    
    return final_preds.numpy()