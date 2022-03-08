from math import cos
import cv2
import numpy as np

im_wid = 256
im_hei = 256

scale1 = np.float32([[1.1, 0, -12.8],
                     [0, 1, 0]])

scale2 = np.float32([[1, 0, 0],
                     [0, 1.1, -12.8]])

scale3 = np.float32([[1.1, 0, -12.8],
                        [0, 1.15, -19.2]])

def create_scale_kernel(dw,dh):
    tx = -im_wid*((dw-1)/2)
    ty = -im_hei*((dh-1)/2)
    return np.float32([[dw, 0, tx],
                       [0, dh, ty]])


def create_skew_kernel(alpha1, alpha2):
    a2 = alpha1 * 2*np.pi/360
    a1 = alpha2 * 2*np.pi/360
    mx = im_wid/2
    my = im_hei/2
    alp1 = np.cos(a1)
    alp2 = np.cos(a2)
    bet1 = np.sin(a1)
    bet2 = np.sin(a2)
    kernel = np.float32([[alp1, bet1, (1-alp1)*mx - bet1*my],
                         [-bet2, alp2, bet2*mx + (1-alp2)*my]])
    return kernel


def lin_trans(source, kernel):
    return cv2.warpAffine(source, kernel, (source.shape[0], source.shape[1]))


#LINUX
# img = cv2.imread('/home/flint/Documents/thesis2022/Python/data/bean.JPG')
#MacOS
img = cv2.imread('/Users/lochuynhquang/Documents/thesis2022/Python/data/bean.JPG')


img = cv2.resize(img, (256,256))

# cv2.imshow('original', img)


# newimg = lin_trans(img, create_scale_kernel(2,2))
newimg = lin_trans(img, create_skew_kernel(30, 0))

cv2.imshow('test', newimg)

# print(create_skew_kernel(0,-30))

cv2.waitKey(0);
cv2.destroyAllWindows();



