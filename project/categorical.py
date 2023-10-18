import torch
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset


class FullCategMLP(pl.LightningModule):
    def __init__(self, input_dim, categ_dims, embedding_dims, hidden_layers, output_dim, lr=1e-3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)

        self.layers.append(nn.Linear(input_dim+sum(embedding_dims), hidden_layers[0]))
        for i in range(1, len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.embedding_layers = nn.ModuleList()
        for categ_dim, embedding_dim in zip(categ_dims, embedding_dims):
            self.embedding_layers.append(nn.Linear(categ_dim, embedding_dim))

        self.loss = nn.MSELoss()

    def forward(self, batch, batch_idx):
        x, categs, y = batch
        for categ, embedding in zip(categs, self.embedding_layers):
            categ = categ.float()
            categ = torch.relu(embedding(categ))
            x = torch.cat([x, categ], dim=1)

        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)

    def embed_categories(self, data):
        return self.embedding_layer(data.float())

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

class FullCategDataset(Dataset):
    def __init__(self, data, categs, target):
        self.data = data
        self.categs = categs
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        categ = []
        for category in self.categs:
            categ.append(category[index])
        y = self.target[index]
        return x, categ, y

    def __len__(self):
        return len(self.data)