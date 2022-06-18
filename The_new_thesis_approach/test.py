from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
import cv2
from image_extractor import preprocess_hsv

# image_dir = 'D:/Thesis_data/Backups/Color_Corrected_512x512/Moldered/image (67).JPG'
# image_dir = 'D:/Thesis_data/Backups/Color_Corrected_512x512/Moldered/image (2).JPG'
image_dir = '/Users/lochuynhquang/Desktop/Color_Corrected_512x512/Plated_PartiallyPurple/image (48).JPG'

image = cv2.imread(image_dir)
image_hsv , cnt, ellipse= preprocess_hsv(image, lut1=None, lut2=None, Contour=True)
image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
plt.imshow(image_rgb)
plt.show()

