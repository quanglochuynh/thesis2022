from asyncore import close_all
import cv2 as cv
import numpy as np

pic = cv.imread('/home/flint/Documents/thesis2022/Python/data/bean.JPG');
pic = cv.resize(pic, (256,256))

cv.imshow('original', pic)






cv.waitKey(0);
cv.destroyAllWindows();