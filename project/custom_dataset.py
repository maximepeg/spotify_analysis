from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset


class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


class CategoricalDataset(Dataset):
    def __init__(self, data, genre, target):
        self.data = data
        self.genre = genre
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        categ = self.genre[index]
        y = self.target[index]
        return x, categ, y

    def __len__(self):
        return len(self.data)
