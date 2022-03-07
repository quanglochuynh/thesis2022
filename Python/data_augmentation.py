from asyncore import close_all
import cv2 as cv
import numpy as np


def lin_trans(source, kernel):
    (wid, hei, dep) = np.shape(source);
    print(wid, hei, dep)
    a = (0,0);
    lut = [a * wid]*hei;
    print(lut)



# pic = cv.imread('/home/flint/Documents/thesis2022/Python/data/bean.JPG');
# pic = cv.resize(pic, (256,512))

# cv.imshow('original', pic)


lin_trans(np.zeros((4,6,3)),[])



# cv.waitKey(0);
# cv.destroyAllWindows();



