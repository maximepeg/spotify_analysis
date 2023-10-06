import pandas as pd
import pytorch_lightning as pl
from itertools import product
from collections import ChainMap


class GridSearch:
    def __init__(self, params, objects):
        """
        :param params: dictionnary of dictionaries of list,
                        each dictionary containing a list of parameter to be put in a grid
                        example : {"model_params":{"lr": [1e-3, 1e-4],
                                                   "dropout": [0.1, 0.2]},
                                   "trainer_params"{"epochs": [50,100,200],
                                                    "strategy": ["ddp", "ddp_spawn"]}
                                   }
        :objects : list of objects to run the grid search on,
                    the number of objects must be equal to the number of dictionnaries in the grid
                    example : [MLP, Trainer]
        """
        self.params = params
        self.objects = objects
        self.grid = self.create_grid()
        self.best_params = None

    def create_grid(self):
        """
        :return: DataFrame containing all the possible combinations of parameters in the grid with a multi index columns
        """
        #check if number of keys in params is equal to number of objects, throw an error if not
        if len(self.params) != len(self.objects):
            raise ValueError("Number of objects must be equal to number of dictionnaries in the grid")

        object_names = list(self.params.keys())
        cols = [(object_names[i], x) for i in range(len(object_names)) for x in self.params[object_names[i]].keys()]
        flat_params = dict(ChainMap(*self.params.values()).items())
        grid = pd.DataFrame(columns=flat_params.keys())
        for i, params in enumerate(product(*flat_params.values())):
            grid.loc[i] = params
        grid.columns = pd.MultiIndex.from_tuples(cols)
        return grid

    def run(self):
        """
        :return: Runs the gridsearch and sets the best_params attribute to a dictionnary of the best parameters
        """
        grid_cols = list(self.grid.columns.get_level_values(0))
        results = pd.DataFrame(columns=grid_cols+["train_loss", "val_loss"])
        for i in range(len(self.grid)):
            params = self.grid.iloc[i].to_dict()


        self.best_params = results.sort_values(by="val_loss").iloc[0].to_dict()
        return results

# test run method of GridSearch class

from pytorch_lightning import Trainer
from mlp import  MLP
grid = GridSearch({"model_params":{"lr": [1e-3, 1e-4],
                                     "dropout": [0.1, 0.2]},
                     "trainer_params":{"max_epochs": [50,100,200]}

                        }, [MLP, Trainer])
results = grid.run()

print

