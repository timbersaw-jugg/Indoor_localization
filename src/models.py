# src/models.py

import torch
import torch.nn as nn

class RSSIAutoencoder(nn.Module):
    def __init__(self, config):
        super(RSSIAutoencoder, self).__init__()
        input_dim = config['model']['input_dim']
        latent_dim = config['model']['latent_dim']
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, latent_dim, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.ConvTranspose1d(32, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = decoded.permute(0, 2, 1)
        return output

class IndoorLocalizer(nn.Module):
    def __init__(self, config, num_classes, pretrained_encoder):
        super(IndoorLocalizer, self).__init__()
        self.input_dim = config['model']['input_dim']
        latent_dim = config['model']['latent_dim']
        self.encoder = pretrained_encoder
        self.lstm = nn.LSTM(
            input_size=latent_dim, hidden_size=latent_dim, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(param)
                    elif 'bias' in name: nn.init.constant_(param, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        assert x.size(1) == self.input_dim, f"Input feature mismatch: expected {self.input_dim}, got {x.size(1)}"
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x