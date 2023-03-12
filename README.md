# Facade segmentation and windows counting
This repo is pet project in computer vision and semantic segmentation. The main aim is design or adaptate CNN to segment windows on different buildings and count them.
I used [CMP Facade Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/) for this task. Dataset contains 606 images with corresponding masks.

## Installation
```
git clone https://github.com/DmitriyKras/facade-segmentation.git
cd your_path/facade-segmentation
```
## Project overview
Config files `prepare_config.json`, `train_config.json` and `evaluate_config.json` contain all important params such as pathes, training settings etc for experiment process. Modify them according to provide information you need. Note: all pathes are relative to project's root folder. `utils` and `models` folders contain py files for training, data preparation and model building. `src` folder contains all files for experiment such as `src/prepare.py`, `src/train.py` and `src/evaluate.py`. `data` folder used to store processed train, val and test pairs.

## Data preparation
CMP Facade dataset has 12 classes with different parts of facade. We are interested in only windows, so `utils/process_masks.py` used to prepare masks for our task. Now all masks contain 255 at windows and 0 at other locations. Processed dataset is available at my Kaggle profile: ... . To generate train, val and test pairs make changes at `prepare_config.json` to provide relative pathes to images and masks folders and destination folder where generated pairs will be stored as JSON files. Then run `src/prepare.py`
