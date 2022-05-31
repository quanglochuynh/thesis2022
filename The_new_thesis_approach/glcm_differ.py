import cv2
import numpy as np
import matplotlib.pyplot as plt 
from image_extractor import preprocess_hsv
from skimage.feature import graycomatrix, graycoprops

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']

image_dir  = 'D:/Thesis_data/Backups/Color_Corrected_512x512/'

level = 32
bins = np.linspace(0, 255+1, level+1)

cons = []
disi = []
ener = []
corr = []

for i in range(len(classes_name)):
    image  = cv2.imread(image_dir + '')
    image_rgb = cv2.cvtColor(preprocess_hsv(image, Contour=False), cv2.COLOR_HSV2RGB)
    all_glcm = []

    for c in range(3):
        digitize = np.digitize(image_rgb[:,:,c], bins) - 1
        glcm = graycomatrix(digitize, [5,7,9,11], [0, np.pi/4, np.pi/2, 3*np.pi/4], level, True, False)
        glcm_cons = graycoprops(glcm, 'contrast')
        glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')
        glcm_energy = graycoprops(glcm, 'ASM')
        glcm_correlation = graycoprops(glcm, 'correlation')
        glcm_homogeneity = graycoprops(glcm, 'homogeneity')
        