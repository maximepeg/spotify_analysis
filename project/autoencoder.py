import torch
import pytorch_lightning as pl
from torch import nn


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_layers, output_dim, lr=1e-3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.dropout = nn.Dropout(dropout)

        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)

    def common_step(self, batch, batch_idx):
        x, _ = batch
        # y = y.unsqueeze(1).float()
        x=x.float()

        out = self(x)

        loss = self.loss(out, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class DAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_layers, output_dim, noise_mean=0.1, noise_std=0.1, lr=1e-3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.dropout = nn.Dropout(dropout)
        self.noise_mean=noise_mean
        self.noise_std=noise_std

        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)

    def common_step(self, batch, batch_idx):
        x, _ = batch
        # y = y.unsqueeze(1).float()
        x = x.float()

        out = self(x)

        loss = self.loss(out, x)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        noisy_x = x.float() + torch.randn_like(x) * self.noise_std+self.noise_mean
        out = self(noisy_x)
        loss = self.loss(out, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)