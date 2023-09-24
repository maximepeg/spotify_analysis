from typing import Any, Optional

import torch
import pytorch_lightning as pl
from torch import nn


class MLP(pl.LightningModule):
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
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)

    def common_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()
        x=x.float()

        logits = self(x)

        loss = self.loss(logits, y)
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


# categorical embedding

class CategMLP(pl.LightningModule):
    def __init__(self, input_dim, embedding_dim, hidden_layers, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.embedding_layer = nn.Embedding(embedding_dim, hidden_layers[0])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        categ, x = x
        categ = self.embedding_layer(categ)
        x = torch.cat((categ, x), dim=1)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
