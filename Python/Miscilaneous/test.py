import cv2
import numpy as np

# di = '/Users/lochuynhquang/Documents/thesis2022/Python/data/bean2.JPG'
di = 'C:/Users/quang/Documents/thesis2022/Python/data/bean2.JPG'
# di ='/home/flint/Documents/thesis2022/Python/data/bean2.JPG'
# image = cv2.imread(di + 'image (1).JPG')
image = cv2.imread(di)
image = cv2.resize(image, (256,256))
kernel1 = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])

kernel2 = np.array([[-1/2, 1, -1/2],
                    [-1/2, 1, -1/2],
                    [-1/2, 1, -1/2]])
edge_detect = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

cv2.imshow('Identity', edge_detect)
# cv2.imwrite('/Users/lochuynhquang/Documents/thesis2022/Python/data/bean2_ED.JPG' ,edge_detect)

print(type(image[1][1][1]))
# print(np.shape(image))

cv2.waitKey(0) 
cv2.destroyAllWindows()


