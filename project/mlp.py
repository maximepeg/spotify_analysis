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
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
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
    def __init__(self, input_dim, categ_dim, embedding_dim, hidden_layers, output_dim, lr=1e-3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.layers.append(nn.Linear(input_dim+embedding_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.embedding_layer = nn.Linear(categ_dim, embedding_dim)
        self.loss = nn.MSELoss()

    def forward(self, batch, batch_idx):
        x, categ, y = batch
        categ = self.embedding_layer(categ.float())
        x = torch.cat((categ, x), dim=1)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)

    def embed_categories(self, batch, batch_idx):
        x, categ, y = batch
        categ = self.embedding_layer(categ.float())
        return categ

    def embed_categories(self, data):
        return self.embedding_layer(data)


    def common_step(self, batch, batch_idx):
        _, _, y = batch
        y = y.unsqueeze(1).float()

        out = self(batch, batch_idx)
        loss = self.loss(out, y)
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


class AttentionMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_layers, output_dim, lr=1e-3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.attentions = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.attentions.append(nn.MultiheadAttention(embed_dim=hidden_layers[i + 1], num_heads=1))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        x = self.layers[0](x)
        for layer,  attention in zip(self.layers[1:-1], self.attentions):
            x = torch.relu(layer(x))
            x = self.dropout(x)
            x = attention(x, x, x)[0]

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
