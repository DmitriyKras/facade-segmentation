import json
import os
import sys
import tensorflow as tf
import timeit 



with open("train_config.json", "r") as f:  # load config file
    config = json.load(f)

root = os.getcwd()  # get path to root

utils_path = os.path.join(root, "utils/")  # get path to utils
sys.path.append(utils_path)  # append to sys path
from utils import DataGenerator, FocalLoss   # import classes from utils

models_path = os.path.join(root, "models/")  # get path to models
sys.path.append(models_path)  # append to sys path
from U_Net import DenseUnet  # import Unet model

# load train and val pairs, set up generators
print("Loading train and val pairs... \n")
train_path = os.path.join(root, config["train_pairs_path"])
val_path = os.path.join(root, config["val_pairs_path"])
with open(train_path, "r") as f:  # load train pairs
    train_pairs = json.load(f)
with open(val_path, "r") as f:  # load val pairs
    val_pairs = json.load(f)

train_gen = DataGenerator(train_pairs,
                          tuple(config["input_shape"]),
                          config["batch_size"],
                          config["train_steps"]
                          )
val_gen = DataGenerator(val_pairs,
                        tuple(config["input_shape"]), 
                        config["batch_size"], 
                        config["val_steps"]
                        )

print("Data loaded successfully\n")
# set up metrics
metrics = []
for metric in config["metrics"]:
    if metric == "precision":
        metrics.append(tf.keras.metrics.Precision())
    elif metric == "recall":
        metrics.append(tf.keras.metrics.Recall())
    elif metric == "accuracy":
        metrics.append(metric)

# build and compile model
print("Building and compiling model...\n")
model = DenseUnet(tuple(config["input_shape"]) + (3,))  # build model

if config["loss"] == "focal_binary_crossentropy":
    loss = FocalLoss(0.25, 3)
else:
    loss = config["loss"]

model.compile(
    loss=loss,
    optimizer=config["optimizer"],
    metrics=metrics
    )

# load weiths if needed
if config["load_weights"] == 1:
    model.load_weights(os.path.join(root, config["weights_load_path"]))

# set checkpoint folder if needed
callbacks_list = []
if config["checkpoint"] == 1:
    checkpoint_path = os.path.join(config["checkpoint_folder"],
                                   "Unet-{epoch:02d}.h5")
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, 
                                 verbose=1, period=5)
    callbacks_list = [checkpoint]

print("Model built and compiled successfully\n")
# fit model
print("Starting training...\n")
start = timeit.default_timer()

history = model.fit(
    train_gen,
    steps_per_epoch=config["train_steps"],
    epochs=config["epochs"],
    validation_data=val_gen,
    callbacks=callbacks_list
    )

total_time = timeit.default_timer() - start
print("Training completed\n")
print("Total time: ", total_time, '\n')
# save results
print("Saving results...\n")
log = history.history
log["training_time"] = total_time
log_path = os.path.join(config["train_log_folder"], "train_result.json")
with open(log_path, "w") as f:
    json.dump(log, f)

print("History saved\n")

save_path = config["weights_save_path"]
model.save_weights(save_path)

print("Weights saved\n")
