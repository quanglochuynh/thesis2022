import cv2
import numpy as np
from matplotlib import pyplot as plt

# di = '/Users/lochuynhquang/Documents/thesis2022/Python/data/bean2.JPG'
di  = 'D:/Thesis_data/color_training_img/Brown/image(345).jpg'
di2 = 'D:/Thesis_data/color_training_img/PartiallyPurple/image(345).jpg'
# di ='/home/flint/Documents/thesis2022/Python/data/bean2.JPG'
# image = cv2.imread(di + 'image (1).JPG')
image = cv2.imread(di)
image2 = cv2.imread(di2)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(image2)

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

# print(image)
plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(image2)
plt.show()
# cv2.imshow('Brown2', image2)
# cv2.imwrite('/Users/lochuynhquang/Documents/thesis2022/Python/data/bean2_ED.JPG' ,edge_detect)

# print(type(image[1][1][1]))
# print(np.shape(image))

# cv2.waitKey(0) 
# cv2.destroyAllWindows()


