import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random


def get_img_mask_pairs(img_folder, mask_folder, val_size=0.2):
    images = sorted(os.listdir(img_folder))  # get list of images
    masks = sorted(os.listdir(mask_folder))  # get list of masks
    pairs = [(os.path.join(img_folder, img), os.path.join(mask_folder, mask)) 
             for img, mask in zip(images, masks)]  # form pairs
    return train_test_split(pairs, test_size=val_size)  # split on train/val


class DataGenerator(tf.keras.utils.Sequence):
    
    
    def __init__(self, pairs, input_shape, batch_size=4,
                 steps_per_epoch=100):
        self.pairs = pairs  # store learning params
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.steps_per_epoch = steps_per_epoch  # less for validation
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch
    
    
    def on_epoch_end(self):
        'Shuffle pairs after each epoch'
        random.shuffle(self.pairs)
        
        
    def get_mask(self, path):
        "Load and resize mask from path"
        mask = cv2.imread(path, 0) / 255 # load mask
        mask = cv2.resize(mask, self.input_shape, 
                          interpolation=cv2.INTER_NEAREST)  # resize mask
        return mask.astype(int)
    
    
    def get_image(self, path):
        "Load and resize image from path"
        img = cv2.imread(path, 1)  # load image
        img = cv2.resize(img, self.input_shape)  # resize image
        return img / 255  # return normalized
    
    
    def get_batch(self, batch_pairs):
        "Get batch of input image and output masks"
        images = []  # create empty lists for images and masks
        masks = []
        for pair in batch_pairs:  # iterate over batch pairs
            img = self.get_image(pair[0])  # load image and mask
            mask = self.get_mask(pair[1])
            images.append(img)  # append to lists
            masks.append(mask)
        images = np.array(images)  # convert to arrays
        masks = np.array(masks)
        return images, masks  # return loaded pair
    
    
    def __getitem__(self, index):
        "Get item - batch with index"
        batch_pairs = self.pairs[index * self.batch_size : 
                                 (index + 1) * self.batch_size]  # get batch pairs
        images, masks = self.get_batch(batch_pairs)  # get batch
        return images, masks  # return batch
    
    
class FocalLoss(tf.keras.losses.Loss):
    
    
    def __init__(self, alpha, gamma):
        self.alpha = alpha  # set loss params
        self.gamma = gamma
        super().__init__(name="focal_loss")
        
        
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()  # define small value
        # clip predictions
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  
        # calculate p_t
        p_t = tf.where(tf.math.equal(y_true, 1), y_pred, 1 - y_pred)
        # calculate alpha weighting selector
        alpha = tf.ones_like(y_true) * self.alpha
        alpha = tf.where(tf.math.equal(y_true, 1), 1 - alpha, alpha)
        # calculate focal crossentropy
        fce = -alpha * tf.math.log(p_t) * tf.math.pow(1 - p_t, self.gamma)
        fce = tf.math.reduce_sum(fce, axis=[1,2])  # sum inside batch
        return fce  # mean of batch losses
