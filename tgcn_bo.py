import optuna
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tgcn_model import TGCNModel
from sklearn.model_selection import TimeSeriesSplit

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

def print_best_trial(study, trial):
    best_trial = study.best_trial
    print(f"Finished trial {trial.number} with value: {trial.value}")
    print(f"Best trial so far: {best_trial.number}")
    print(f"Best value so far: {best_trial.value}")
    print(f"Best parameters so far: {best_trial.params}")

def objective(trial):
    gru_units = trial.suggest_categorical('gru_units', [16, 32, 64, 128])
    l1 = trial.suggest_float('l1', 0.001, 1, log=True)
    l2 = trial.suggest_float('l2', 0.001, 1, log=True)
    epochs = trial.suggest_int('epochs', 10, 100)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

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

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[print_best_trial])

best_hyperparameters = study.best_params
print("Best hyperparameters: ", best_hyperparameters)

with open('best_hyperparameters_optuna.pkl', 'wb') as f:
    pickle.dump(best_hyperparameters, f)

with open('study_optuna.pkl', 'wb') as f:
    pickle.dump(study, f)