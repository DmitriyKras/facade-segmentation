import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


path = "masks/cmp_b0013.png"
img = "images/cmp_b0013.jpg"

MASK_FOLDER = "masks"
NEW_MASK_FOLDER = "windows"
"""
mask = cv2.imread(path,0)
seg_mask = np.zeros((mask.shape) + (12,), dtype=int)

codes = [19, 29, 76, 78, 126, 128, 176, 178, 194, 210, 225]

for i, code in enumerate(codes):
    seg_mask[..., i] = (mask == code).astype(int) 

for i in range(12):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    plt.imshow(seg_mask[..., i] * 255, cmap = "gray")
    
"""
mask = cv2.imread(path,0)
seg_mask = np.logical_or(mask == 78, mask == 225).astype(int)
plt.figure()
plt.imshow(seg_mask * 255, cmap = "gray")

masks_list = os.listdir(MASK_FOLDER)
for mask in masks_list:
    seg_mask = cv2.imread(os.path.join(MASK_FOLDER, mask), 0)
    seg_mask = np.logical_or(seg_mask == 78, seg_mask == 225) * 255
    cv2.imwrite(os.path.join(NEW_MASK_FOLDER, mask), seg_mask)
