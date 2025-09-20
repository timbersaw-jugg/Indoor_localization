# src/training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import mlflow
import joblib

from src.models import RSSIAutoencoder, IndoorLocalizer
from src.utils import plot_curves, plot_gradient_descent_progress

def train_one_fold(config, X_train, y_train, X_val, y_val, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nsamples, nsteps, nfeatures = X_train.shape
    X_train_reshaped = X_train.reshape(-1, nfeatures)
    scaler = StandardScaler().fit(X_train_reshaped)
    
    scaler_path = os.path.join(config['paths']['model_dir'], f'scaler_fold_{fold}.joblib')
    joblib.dump(scaler, scaler_path)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(nsamples, nsteps, nfeatures)
    X_val_scaled = scaler.transform(X_val.reshape(-1, nfeatures)).reshape(X_val.shape)
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_scaled).float(), torch.from_numpy(y_train).long()),
        batch_size=config['training']['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_scaled).float(), torch.from_numpy(y_val).long()),
        batch_size=config['training']['batch_size']
    )

    print("Pre-training Autoencoder...")
    ae_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_scaled).float()), batch_size=128, shuffle=True)
    autoencoder = RSSIAutoencoder(config).to(device)
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    for _ in range(25):
        autoencoder.train()
        for (inputs,) in ae_loader:
            inputs = inputs.to(device)
            noisy_inputs = inputs + 0.05 * torch.randn_like(inputs)
            reconstructions, _ = autoencoder(noisy_inputs)
            loss = ae_criterion(reconstructions, inputs)
            ae_optimizer.zero_grad(); loss.backward(); ae_optimizer.step()
    
    pretrained_encoder = autoencoder.encoder
    for param in pretrained_encoder.parameters():
        param.requires_grad = True

    num_classes = len(np.unique(np.concatenate((y_train, y_val))))
    model = IndoorLocalizer(config, num_classes, pretrained_encoder).to(device)
    
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': config['training']['learning_rate'] * 0.5},
        {'params': head_params, 'lr': config['training']['learning_rate']}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4)

    best_val_loss = float('inf'); epochs_no_improve = 0
    history = {'train_losses': [], 'val_losses': [], 'val_accuracies': []}
    tracked_weights = []

    print(f"Training IndoorLocalizer for Fold {fold}...")
    for epoch in range(config['training']['epochs']):
        model.train(); total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_losses'].append(avg_train_loss)
        tracked_weights.append(model.fc1.weight[0][0].detach().cpu().item())
        
        model.eval(); total_val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_val_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item(); total += yb.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        history['val_losses'].append(avg_val_loss); history['val_accuracies'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config['paths']['model_dir'], f'model_fold_{fold}.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['training']['patience']:
                print("Early stopping triggered."); break
    
    model.load_state_dict(torch.load(os.path.join(config['paths']['model_dir'], f'model_fold_{fold}.pth')))
    plot_gradient_descent_progress(tracked_weights, history['train_losses'], config, fold=fold)
    plot_curves(history, config, title_prefix=f"Fold {fold}", filename=f"curves_fold_{fold}.png")
    return model

def train_kfold_ensemble(config, X_train_val, y_train_val):
    _, counts = np.unique(y_train_val, return_counts=True)
    n_splits = int(max(3, min(config['training']['k_folds'], counts.min())))
    print(f'Using n_splits={n_splits} (based on class distribution)')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])
    trained_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), 1):
        with mlflow.start_run(run_name=f"Fold_{fold}", nested=True):
            print(f"\n===== FOLD {fold} / {n_splits} =====")
            X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
            y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
            model = train_one_fold(config, X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold)
            trained_models.append(model)
            
            model.eval()
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val_fold).float(), torch.from_numpy(y_val_fold).long()),
                batch_size=config['training']['batch_size']
            )
            y_pred, y_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb_reshaped = xb.numpy().reshape(-1, xb.shape[2])
                    scaler = joblib.load(os.path.join(config['paths']['model_dir'], f'scaler_fold_{fold}.joblib'))
                    xb_scaled = torch.from_numpy(scaler.transform(xb_reshaped).reshape(xb.shape)).float()
                    out = model(xb_scaled.to('cuda' if torch.cuda.is_available() else 'cpu'))
                    y_pred.extend(out.argmax(dim=1).cpu().numpy())
                    y_true.extend(yb.cpu().numpy())
            
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            mlflow.log_metric("final_val_f1_macro", f1)
            print(f"Fold {fold} Final Validation F1 (Macro): {f1:.4f}")

    return trained_models