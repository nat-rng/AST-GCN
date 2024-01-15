import os
import optuna
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from tgcn_model import TGCNModel
from sklearn.model_selection import TimeSeriesSplit
from optuna.pruners import MedianPruner

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
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial so far: {best_trial.number}")
    print(f"Best value so far: {best_trial.value}")
    print(f"Best parameters so far: {best_trial.params}")

def r_squared(y_true, y_pred):
    ss_res =  K.sum(K.square(y_true - y_pred)) 
    ss_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - ss_res/(ss_tot + K.epsilon()))

lambda_loss = 0.0015

def get_loss_function(model):
    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, num_nodes])
        y_pred = tf.reshape(y_pred, [-1, num_nodes])
        mse_term = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        l2_reg_term = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'kernel' in v.name])
        total_loss = mse_term + lambda_loss * l2_reg_term
        return total_loss
    return loss_function

class HyperparametersLogger(Callback):
    def __init__(self, hyperparameters, trial_number):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.trial_number = trial_number

    def on_train_begin(self, logs=None):
        print(f"Trial number: {self.trial_number}, "
              f"Training started with the following hyperparameters: {self.hyperparameters}")

    def on_epoch_end(self, logs=None):
        print(f"Trial number: {self.trial_number}, "
              f"Hyperparameters used in epoch: {self.hyperparameters}")



def objective(trial):
    trial_number = trial.number
    num_gpus = 2
    gpu_id = trial.number % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gru_units = trial.suggest_categorical('gru_units', [16, 32, 64, 128])
    l1 = trial.suggest_float('l1', 0.001, 1, log=True)
    l2 = trial.suggest_float('l2', 0.001, 1, log=True)
    # epochs = trial.suggest_categorical('epochs', [10, 15, 20, 30, 40, 50])
    epochs = 30
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    hyperparameters = trial.params
    val_losses = []
    tscv = TimeSeriesSplit(n_splits=3)
    for train_indices, test_indices in tscv.split(trainX):
        x_train, y_train = trainX[train_indices], trainY[train_indices]
        x_val, y_val = trainX[test_indices], trainY[test_indices]

        model = TGCNModel(num_nodes, gru_units, adj, pre_len, l1, l2)
        loss = get_loss_function(model)
        model.compile(optimizer='adam', loss=loss, metrics=['mae','mse','mape', r_squared])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val), verbose=2, 
                            callbacks=[HyperparametersLogger(hyperparameters, trial_number)])

        val_losses.append(history.history['val_loss'][-1])
    return np.mean(val_losses)

pruner = MedianPruner()
study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[print_best_trial], n_jobs=2)

best_hyperparameters = study.best_params
print("Best hyperparameters: ", best_hyperparameters)

with open('best_hyperparameters_optuna.pkl', 'wb') as f:
    pickle.dump(best_hyperparameters, f)

with open('study_optuna.pkl', 'wb') as f:
    pickle.dump(study, f)