import os
import GPUtil
import pickle
import numpy as np
import pandas as pd
import ray
from tgcn_model import TGCNModel
import tensorflow as tf
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from sklearn.model_selection import TimeSeriesSplit

num_cpus = os.cpu_count()
num_gpus = len(GPUtil.getGPUs())

resources_per_trial = {
    "cpu": 2,
    "gpu": 0.5 if num_gpus > 0 else 0
}

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

with open("data/timestep_24/testX_timestep_24_20240108.pkl", 'rb') as file:
    testX_loaded = pickle.load(file)

with open("data/timestep_24/testY_timestep_24_20240108.pkl", 'rb') as file:
    testY_loaded = pickle.load(file)

road_adj = pd.read_excel("road_connection.xlsx")
adj = np.mat(road_adj)

trainX = np.array(trainX_loaded)
trainY = np.array(trainY_loaded)
testX = np.array(testX_loaded)
testY = np.array(testY_loaded)

num_nodes = adj.shape[0]
pre_len = 12

def chunk_data(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

chunk_size = 1000  
trainX_chunks = chunk_data(trainX, chunk_size)
trainY_chunks = chunk_data(trainY, chunk_size)

def build_and_train_model(config, x_train, y_train, x_val, y_val, checkpoint_dir=None):
    gru_units = config["gru_units"]
    l1 = config["l1"]
    l2 = config["l2"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    model = TGCNModel(num_nodes, gru_units, adj, pre_len, l1, l2)
    model.compile(optimizer='adam', loss='mse')

    if checkpoint_dir:
        model_path = os.path.join(checkpoint_dir, "model.h5")
        model.load_weights(model_path)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), verbose=0)

    if checkpoint_dir:
        model.save_weights(os.path.join(checkpoint_dir, "model.h5"))

    return model, history.history

def tune_tgcn(config, trainX_chunk_ids, trainY_chunk_ids):
    tscv = TimeSeriesSplit(n_splits=5)
    val_losses = []

    for chunk_id in range(len(trainX_chunk_ids)):
        trainX_chunk = ray.get(trainX_chunk_ids[chunk_id])
        trainY_chunk = ray.get(trainY_chunk_ids[chunk_id])
        
        for train_indices, test_indices in tscv.split(trainX_chunk):
            x_train, y_train = trainX_chunk[train_indices], trainY_chunk[train_indices]
            x_val, y_val = trainX_chunk[test_indices], trainY_chunk[test_indices]

            model, history = build_and_train_model(config, x_train, y_train, x_val, y_val)
            val_losses.append(history['val_loss'][-1])

    tune.report(val_loss=np.mean(val_losses))

ray.init(num_cpus=num_cpus, num_gpus=num_gpus) 

try:
    trainX_chunk_ids = [ray.put(chunk) for chunk in trainX_chunks]
    trainY_chunk_ids = [ray.put(chunk) for chunk in trainY_chunks]

    config = {
        "gru_units": tune.randint(16, 256),
        "l1": tune.loguniform(0.001, 1),
        "l2": tune.loguniform(0.001, 1),
        "epochs": tune.randint(10, 100),
        "batch_size": tune.randint(16, 128)
    }

    bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=50, reduction_factor=4)
    bohb_search = TuneBOHB()

    analysis = tune.run(
        tune.with_parameters(tune_tgcn, trainX_chunk_ids=trainX_chunk_ids, trainY_chunk_ids=trainY_chunk_ids),
        name="tgcn_bohb_optimization",
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=10,
        scheduler=bohb_scheduler,
        search_alg=bohb_search,
        resources_per_trial=resources_per_trial
    )

    best_hps = analysis.get_best_config(metric="val_loss", mode="min")
    with open('best_hyperparameters_bohb_gpu.pkl', 'wb') as f:
        pickle.dump(best_hps, f)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    ray.shutdown()