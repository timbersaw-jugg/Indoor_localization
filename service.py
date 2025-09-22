# # service.py

# import bentoml
# import numpy as np
# import torch
# from bentoml.io import NumpyNdarray

# BENTO_MODEL_TAG = "indoor_localization_ensemble:latest"

# # 1. Load the BentoML model from the local store
# bento_model = bentoml.models.get(BENTO_MODEL_TAG)

# # 2. Create a class to define our prediction service
# @bentoml.service
# class IndoorLocalizationClassifier:
    
#     # 3. Load the models and scalers into the class instance
#     # The 'bento' dictionary is automatically passed here by the server
#     bento = bento_model
    
#     def __init__(self):
#         self.models_dict = self.bento.custom_objects["models"]
#         self.scalers_dict = self.bento.custom_objects["scalers"]
#         self.device = torch.device("cpu")
#         print("IndoorLocalizationClassifier service initialized successfully.")
    
#     # 4. Define the API endpoint for prediction
#     @bentoml.api
#     def predict(self, input_sequence: np.ndarray) -> np.ndarray:
#         all_logits = []
        
#         for i in range(1, len(self.models_dict) + 1):
#             model = self.models_dict[f"model_fold_{i}"]
#             scaler = self.scalers_dict[f"scaler_fold_{i}"]
            
#             model.to(self.device)
#             model.eval()
            
#             nsamples, nsteps, nfeatures = input_sequence.shape
#             input_reshaped = input_sequence.reshape(-1, nfeatures)
#             input_scaled = scaler.transform(input_reshaped).reshape(nsamples, nsteps, nfeatures)
#             input_tensor = torch.from_numpy(input_scaled).float().to(self.device)
            
#             with torch.no_grad():
#                 outputs = model(input_tensor)
#                 all_logits.append(outputs.cpu())
        
#         avg_logits = torch.stack(all_logits).mean(dim=0)
#         final_preds = avg_logits.argmax(dim=1)
        
#         return final_preds.numpy()

import bentoml
import numpy as np
import torch

BENTO_MODEL_TAG = "indoor_localization_ensemble:yy6il7uxvcpj5lrk"

# Load the BentoML model from the local store
bento_model = bentoml.models.get(BENTO_MODEL_TAG)

@bentoml.service
class IndoorLocalizationClassifier:
    def __init__(self):
        self.models_dict = bento_model.custom_objects["models"]
        self.scalers_dict = bento_model.custom_objects["scalers"]
        self.device = torch.device("cpu")
        print("IndoorLocalizationClassifier service initialized successfully.")
    
    @bentoml.api
    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        all_logits = []
        
        for i in range(1, len(self.models_dict) + 1):
            model = self.models_dict[f"model_fold_{i}"]
            scaler = self.scalers_dict[f"scaler_fold_{i}"]
            
            model.to(self.device)
            model.eval()
            
            nsamples, nsteps, nfeatures = input_sequence.shape
            input_reshaped = input_sequence.reshape(-1, nfeatures)
            input_scaled = scaler.transform(input_reshaped).reshape(nsamples, nsteps, nfeatures)
            input_tensor = torch.from_numpy(input_scaled).float().to(self.device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                all_logits.append(outputs.cpu())
        
        avg_logits = torch.stack(all_logits).mean(dim=0)
        final_preds = avg_logits.argmax(dim=1)
        
        return final_preds.numpy()