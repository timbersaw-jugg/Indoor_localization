# src/evaluation.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib

def ensemble_predict(models, scalers, dataloader, device):
    """
    Performs ensemble prediction, ensuring each model uses its corresponding
    scaler to process the test data before inference.
    """
    all_logits = []
    
    if len(models) != len(scalers):
        raise ValueError(f"Mismatch between number of models ({len(models)}) and scalers ({len(scalers)})")

    for i, model in enumerate(models):
        scaler = scalers[i]
        model.to(device)
        model.eval()
        
        logits_fold = []
        with torch.no_grad():
            for (inputs,) in dataloader:
                # Scale the test data batch before sending it to the model
                inputs_numpy = inputs.numpy()
                nsamples, nsteps, nfeatures = inputs_numpy.shape
                inputs_reshaped = inputs_numpy.reshape(-1, nfeatures)
                inputs_scaled = scaler.transform(inputs_reshaped).reshape(nsamples, nsteps, nfeatures)
                
                inputs_tensor = torch.from_numpy(inputs_scaled).float().to(device)

                outputs = model(inputs_tensor)
                logits_fold.append(outputs.cpu())
                
        all_logits.append(torch.cat(logits_fold, dim=0))

    # Average the logits across all models (soft voting)
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = avg_logits.argmax(dim=1)
    return preds.numpy()