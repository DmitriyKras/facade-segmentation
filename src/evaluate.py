import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, roc_auc_score

with open("evaluate_config.json", "r") as f:  # load config file
    config = json.load(f)

root = os.getcwd()  # get path to root

models_path = os.path.join(root, "models/")  # get path to models
sys.path.append(models_path)  # append to sys path
from U_Net import DenseUnet  # import Unet model

# build model
model = DenseUnet(tuple(config["input_shape"]) + (3,))  # build model

# load weiths for evaluating
model.load_weights(os.path.join(root, config["weights_path"]))

# load evaluating pairs
eval_path = os.path.join(root, config["pairs_path"])  # get path for eval pairs
with open(eval_path, "r") as f:  # load eval pairs
    eval_pairs = json.load(f)

y_pred = []  # init empty lists for collecting
y_true = []

start = timeit.default_timer()

for pair in tqdm(eval_pairs, total=len(eval_pairs),
                 desc="Collecting predictions..."):
    img = cv2.imread(pair[0])  # load image
    # resize image to input shape
    img = cv2.resize(img, tuple(config["input_shape"])) / 255 
    pred = model.predict(np.expand_dims(img, axis=0),
                         verbose=0)  # get prediction
    mask = cv2.imread(pair[1], 0) / 255  # load mask
    # resize mask to input shape
    mask = cv2.resize(mask, tuple(config["input_shape"]),
                      interpolation=cv2.INTER_NEAREST)
    # stack predictions and masks
    y_true.append(mask)
    y_pred.append(np.squeeze(pred))

# convert to ndarray and flat
y_true = np.array(y_true).flatten().astype(int)
y_pred = np.array(y_pred).flatten()

metrics = {}
# compute precision, recall and f-score
metrics["precision"] = precision_score(y_true, (y_pred > config["threshold"]).astype(int))
metrics["recall"] = recall_score(y_true, (y_pred > config["threshold"]).astype(int))
metrics["f-score"] = f1_score(y_true, (y_pred > config["threshold"]).astype(int))


#compute AP and AUC
metrics["AP-score"] = average_precision_score(y_true, y_pred)
metrics["ROC-AUC-score"] = roc_auc_score(y_true, y_pred)

#compute PR and ROC curves if needed
eval_log_path = os.path.join(root, config["evaluate_log_folder"])
if config["plot_PR"] == 1:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(precision, recall, linewidth=2, color="red")
    plt.title("Precision-recall curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.savefig(os.path.join(eval_log_path, "PR_curve.png"))

if config["plot_ROC"] == 1:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, color="red")
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(os.path.join(eval_log_path, "ROC_curve.png"))

total_time = timeit.default_timer() - start
print("Total evaluation time:", total_time)

metrics["eval_time"] = total_time

with open(os.path.join(eval_log_path, "eval_log.json"), "w") as f:
    json.dump(metrics, f)
