import cv2
import numpy as np

image = cv2.imread('/Users/lochuynhquang/Documents/thesis2022/Python/data/comga.jpeg')

kernel1 = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])

edge_detect = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
# cv2.imshow('Original', image)
cv2.imshow('Identity', edge_detect)


cv2.waitKey(0) 
cv2.destroyAllWindows()