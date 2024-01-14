import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from tgcn_model import TGCNModel
from tensorflow.keras.callbacks import EarlyStopping

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load data
with open("data/timestep_24/trainX_timestep_24_20240108.pkl", 'rb') as file:
    trainX = pickle.load(file)

with open("data/timestep_24/trainY_timestep_24_20240108.pkl", 'rb') as file:
    trainY = pickle.load(file)

with open("data/timestep_24/testX_timestep_24_20240108.pkl", 'rb') as file:
    testX = pickle.load(file)

with open("data/timestep_24/testY_timestep_24_20240108.pkl", 'rb') as file:
    testY = pickle.load(file)

road_adj = pd.read_excel("road_connection.xlsx")
adj = np.mat(road_adj)

num_nodes = adj.shape[0]
pre_len = 12

def build_model(hp):
    gru_units = hp.Choice('gru_units', values=[16, 32, 64, 128])
    l1 = hp.Float('l1', min_value=0.001, max_value=1, sampling='log')
    l2 = hp.Float('l2', min_value=0.001, max_value=1, sampling='log')
    batch_size = hp.Choice('batch_size', values=[16, 32, 64, 128])
    epochs = hp.Int('epochs', min_value=10, max_value=100, step=10)

    model = TGCNModel(num_nodes, gru_units, adj, pre_len, l1, l2)
    model.compile(optimizer='adam', loss='mse')

    return model, batch_size, epochs

tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name='tgcn_model_tuning'
)

tscv = TimeSeriesSplit(n_splits=3)
for trial in tuner.oracle.create_trials(max_trials=10):
    hp = trial.hyperparameters
    batch_size = hp.get('batch_size')
    epochs = hp.get('epochs')

    val_losses = []

    for train_indices, val_indices in tscv.split(trainX):
        x_train, y_train = trainX[train_indices], trainY[train_indices]
        x_val, y_val = trainX[val_indices], trainY[val_indices]

        model = build_model(hp)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=0)
        val_losses.append(min(history.history['val_loss']))

    avg_val_loss = np.mean(val_losses)
    tuner.oracle.update_trial(trial.trial_id, {'val_loss': avg_val_loss})
    tuner.save_model(trial.trial_id, model)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

with open('best_hyperparameters_keras_tuner.pkl', 'wb') as f:
    pickle.dump(best_hps.values, f)