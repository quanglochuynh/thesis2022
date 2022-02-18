import cv2
import numpy as np

di = '/thesis/no_aug/Grape___Black_rot/'

image = cv2.imread(di + 'image (1).JPG')
# kernel1 = np.array([[-1, -1, -1],
#                     [-1,  8, -1],
#                     [-1, -1, -1]])
# edge_detect = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

# cv2.imshow('Identity', image)

print(type(image))
print(np.shape(image))

cv2.waitKey(0) 
cv2.destroyAllWindows()


