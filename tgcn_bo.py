import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tgcn_model import TGCNModel
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with open("data/timestep_24/trainX_timestep_24_20240108.pkl", 'rb') as file:
    trainX_loaded = pickle.load(file)

with open("data/timestep_24/trainY_timestep_24_20240108.pkl", 'rb') as file:
    trainY_loaded = pickle.load(file)

road_adj = pd.read_excel("road_connection.xlsx")
adj = np.mat(road_adj)

trainX = np.array(trainX_loaded)
trainY = np.array(trainY_loaded)

num_nodes = adj.shape[0]
pre_len = 12

space = [
    Integer(16, 256, name="gru_units"),
    Real(0.001, 1, "log-uniform", name="l1"),
    Real(0.001, 1, "log-uniform", name="l2"),
    Integer(10, 100, name="epochs"),
    Categorical([16, 32, 64, 128], name="batch_size")
]

@use_named_args(space)
def optimize_tgcn(gru_units, l1, l2, epochs, batch_size):
    val_losses = []

    tscv = TimeSeriesSplit(n_splits=3)
    for train_indices, test_indices in tscv.split(trainX):
        x_train, y_train = trainX[train_indices], trainY[train_indices]
        x_val, y_val = trainX[test_indices], trainY[test_indices]

        model = TGCNModel(num_nodes, gru_units, adj, pre_len, l1, l2)
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val), verbose=0)

        val_losses.append(history.history['val_loss'][-1])

    return np.mean(val_losses)

res = gp_minimize(optimize_tgcn, space, n_calls=50, random_state=42)
best_hyperparameters = {dim.name: res.x[i] for i, dim in enumerate(space)}
print("Best hyperparameters: ", best_hyperparameters)

with open('best_hyperparameters_bayesian_opt.pkl', 'wb') as f:
    pickle.dump(best_hyperparameters, f)