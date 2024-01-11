from tgcn_bohb import TGCNModel
import pickle
import os
import numpy as np
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit

with open("/data/timestep_24/trainX_timestep_24_20240108.pkl", 'rb') as file:
    trainX_loaded = pickle.load(file)

with open("/data/timestep_24/trainY_timestep_24_20240108.pkl", 'rb') as file:
    trainY_loaded = pickle.load(file)

with open("/data/timestep_24/testX_timestep_24_20240108.pkl", 'rb') as file:
    testX_loaded = pickle.load(file)

with open("/data/timestep_24/testY_timestep_24_20240108.pkl", 'rb') as file:
    testY_loaded = pickle.load(file)

road_adj = pd.read_excel("/data/road_connection.xlsx")
adj = np.mat(road_adj)

trainX = np.array(trainX_loaded)
trainY = np.array(trainY_loaded)
testX = np.array(testX_loaded)
testY = np.array(testY_loaded)

num_nodes = adj.shape[0]
pre_len = 12

def build_and_train_model(hp, x_train, y_train, x_val, y_val):
    gru_units = hp.Int('gru_units', min_value=16, max_value=256, step=16)
    l1 = hp.Float('l1', min_value=0.0001, max_value=0.1, sampling='log')
    l2 = hp.Float('l2', min_value=0.0001, max_value=0.1, sampling='log')
    epochs = hp.Int('epochs', min_value=10, max_value=300, step=5)
    batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

    model = TGCNModel(num_nodes, gru_units, adj, pre_len, l1, l2)
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), verbose=0)

    return model, history.history


def save_history(trial_id, hp, history, filename='all_trials_history.pkl'):
    all_data = {}
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            all_data = pickle.load(f)
    all_data[trial_id] = {
        'hyperparameters': hp.values,
        'history': history
    }
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f)

class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, splits):
        hp = trial.hyperparameters
        val_losses = []
        full_histories = []
        for train_indices, test_indices in splits:
            x_train, y_train = x[train_indices], y[train_indices]
            x_val, y_val = x[test_indices], y[test_indices]
            model, history = build_and_train_model(hp, x_train, y_train, x_val, y_val)
            val_losses.append(history['val_loss'])
            full_histories.append(history)

        avg_val_loss = np.mean([h[-1] for h in val_losses])
        self.oracle.update_trial(trial.trial_id, {'val_loss': avg_val_loss})
        self.save_model(trial.trial_id, model)
        save_history(trial.trial_id, hp, full_histories)

tscv = TimeSeriesSplit(n_splits=5)

tuner = CVTuner(
    oracle=kt.oracles.Hyperband(
        objective=kt.Objective("val_loss", direction="min"),
        max_epochs=300,
        hyperband_iterations=2),
    hypermodel=build_and_train_model,
    directory='my_dir',
    project_name='tgcn_hyperparam_opt',
    overwrite=True
)

tuner.search(trainX, trainY, splits=tscv.split(trainX))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

with open('best_hyperparameters.pkl', 'wb') as f:
    pickle.dump(best_hps.values, f)