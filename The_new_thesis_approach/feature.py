import cv2
import numpy as np
import matplotlib.pyplot as plt

# cwd = os.getcwd()
image_dir = 'D:/Thesis_data/Backups/Color_Corrected_512x512/Compartmentalized_Purple/image (6).JPG'

image = cv2.imread(image_dir)
print(np.shape(image))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()