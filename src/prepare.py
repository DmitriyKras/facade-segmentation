import json
import os
import sys
from sklearn.model_selection import train_test_split

root = os.getcwd()  # get path to root

with open("prepare_config.json", "r") as f:  # load config file
    config = json.load(f)
    

utils_path = os.path.join(root, "utils/")  # get path to utils

sys.path.append(utils_path)  # append to sys path

from utils import get_img_mask_pairs  # import function from utils

img_folder = config["images_folder"]  # get pathes to img
mask_folder = config["masks_folder"]  # and masks
# get train and val pairs
train_pairs, val_pairs = get_img_mask_pairs(img_folder,
                                            mask_folder, 
                                            val_size=0.2)
# get train and test pairs
train_pairs, test_pairs = train_test_split(train_pairs, test_size=0.1)
# get destination pathes for train, val and test
train_path = os.path.join(root, config["train_pairs_path"])
val_path = os.path.join(root, config["val_pairs_path"])
test_path = os.path.join(root, config["test_pairs_path"])
# write json files
with open(train_path, "w") as f:
    f.write(json.dumps(train_pairs))
with open(val_path, "w") as f:
    f.write(json.dumps(val_pairs))
with open(test_path, "w") as f:
    f.write(json.dumps(test_pairs))

print("Data preparing completed")
