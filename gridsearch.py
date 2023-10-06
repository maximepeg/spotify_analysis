import numpy as np
import pandas as pd
from project.mlp import MLP, CategMLP
from project.custom_dataset import CustomDataset, CategoricalDataset
from sklearn.preprocessing import scale, MultiLabelBinarizer
from tqdm import tqdm

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
df = pd.read_csv("data/playlist_2010to2022.csv").dropna()
df = df.drop(columns=['playlist_url', 'track_id', 'artist_id'], axis=1)
mlb = MultiLabelBinarizer(sparse_output=True)
genres = pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(df.artist_genres.str.strip("[]").str.split(', ')),
                                           columns=mlb.classes_, index=df.index).drop("", axis=1)
genres.columns = genres.columns.str.strip("'")
df = df.drop(columns=['artist_genres'], axis=1)
ohe_year = pd.get_dummies(df.year).astype(int)
target = df.pop("track_popularity")
df = df.drop(columns=['track_name', 'year', 'artist_name', 'artist_popularity', 'album'], axis=1)
df.columns = df.columns.astype(str)
#split in train and test 0.8 t 0.2

full = pd.concat([df, genres, ohe_year], axis=1)
full.columns = full.columns.astype(str)

train = full.sample(frac=0.8, random_state=0)
test = full.drop(train.index)
train_idx = train.index
train_target = target[train.index].values
test_target = target[test.index].values

train, test = scale(train), scale(test)


feature_year_no_genre = pd.concat([df, ohe_year], axis=1)
feature_year_no_genre.columns = feature_year_no_genre.columns.astype(str)

train_no_genre = feature_year_no_genre.loc[train_idx].values
test_no_genre = feature_year_no_genre.drop(train_idx).values
train_no_genre, test_no_genre = scale(train_no_genre), scale(test_no_genre)


# create test and train dataloader from the dataset
train, test = train_no_genre.astype(np.float32), test_no_genre.astype(np.float32)
train_dataset = CategoricalDataset(data=train, target=train_target, genre=genres.loc[train_idx].values)
test_dataset =  CategoricalDataset(data=test, target=test_target, genre=genres.drop(train_idx).values)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


# initialize mlp and create a gridsearch to find the best parameters recording test and train error for each combination of parameters
lr = [0.001, 0.01, 0.1]
hidden_layer_width = [32, 64, 128]
hidden_layer_depth = [1, 2, 4, 8]
embedding_dim = [16, 32, 64]
epochs = [100]
dropout = [0.2]
grid = np.array(np.meshgrid(lr, hidden_layer_width, hidden_layer_depth, embedding_dim, epochs, dropout)).T.reshape(-1, 6)
grid = pd.DataFrame(grid, columns=["lr", "hidden_layer_width", "hidden_layer_depth", "embedding_dim", "epochs", "dropout"])
grid["train_error"] = np.nan
grid["test_error"] = np.nan
grid[["hidden_layer_width", "hidden_layer_depth", "embedding_dim", "epochs"]] = grid[["hidden_layer_width", "hidden_layer_depth", "embedding_dim", "epochs"]].astype(int)

# go through the grid and train the model for each combination of parameters
for i in tqdm(range(len(grid))):
    layers = [grid.hidden_layer_width[i]]*grid.hidden_layer_depth[i]
    model = CategMLP(input_dim=train.shape[1], output_dim=1, categ_dim=genres.shape[1], embedding_dim=grid.embedding_dim[i], hidden_layers=layers,
                     lr=grid.lr[i], dropout=grid.dropout[i], )
    trainer = Trainer(max_epochs=int(grid.epochs[i]), enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_dataloader, test_dataloader)
    grid.train_error[i] = trainer.callback_metrics["train_loss"].item()
    grid.test_error[i] = trainer.callback_metrics["val_loss"].item()

print(grid.sort_values(by="test_error"))
print(grid.sort_values(by="test_error").iloc[0])
#
# n = 20
# # take n random points in the grid for a random search
# random_grid = grid.sample(n=n, random_state=0).reset_index(drop=True)
# #%%
#
# # go through the grid of randomly chosen points and train the model for each combination of parameters
# for i in tqdm(range(len(random_grid))):
#     layers = [random_grid.hidden_layer_width[i]]*random_grid.hidden_layer_depth[i]
#     model = CategMLP(input_dim=train.shape[1], output_dim=1, categ_dim=genres.shape[1], embedding_dim=random_grid.embedding_dim[i], hidden_layers=layers,
#                      lr=random_grid.lr[i], dropout=random_grid.dropout[i], )
#     trainer = Trainer(max_epochs=int(random_grid.epochs[i]), enable_progress_bar=False, enable_model_summary=False)
#     trainer.fit(model, train_dataloader, test_dataloader)
#     random_grid.train_error[i] = trainer.callback_metrics["train_loss"].item()
#     random_grid.test_error[i] = trainer.callback_metrics["val_loss"].item()
#
# print(random_grid.sort_values(by="test_error"))