from hashlib import new
import cv2
import numpy as np
from numpy import random as rd, uint8

im_wid = 256
im_hei = 256

def curved(x):
    return (1/256)* x * x

def image_correct(img, channel):
    w,h,d = np.shape(img)
    newimg = np.zeros((w,h,d), dtype=uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newimg[i][j] = img[i][j];
            newimg[i][j][channel] = curved(img[i][j][channel])
    return newimg


def random(a,b):
    return np.round(a + rd.random()*(b-a), 2)

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

def create_offset_kernel(dx, dy):
    return np.float32([[1, 0, dx], [0, 1, dy]])

def horizontal_flip_kernel():
    return np.float32([[-1, 0, im_wid], [0, 1, 0]])

def vertical_flip_kernel():
    return np.float32([[1, 0, 0], [0, -1, im_hei]])

def create_rotate_kernel(angle):
    return create_skew_kernel(angle, angle)

def lin_trans(source, kernel):
    return cv2.warpAffine(source, kernel, (source.shape[0], source.shape[1]))

func = [create_scale_kernel, create_skew_kernel, create_offset_kernel, horizontal_flip_kernel, vertical_flip_kernel, create_rotate_kernel]






#LINUX
# img = cv2.imread('/home/flint/Documents/thesis2022/Python/data/bean.JPG')
#MacOS
img = cv2.imread('/Users/lochuynhquang/Documents/thesis2022/Python/data/bean.JPG')


img = cv2.resize(img, (im_wid,im_hei))

img = image_correct(img, 1)
# print(img[100][100][2])
# print(type(img[1][1][1]))
cv2.imshow('original', img)

# a = [[100, 128, 276], [0, 256, 0]]

# a = map(curved, a[:][1])
# print(list(a))
# newimg = lin_trans(img, create_scale_kernel(2,1))
# newimg = lin_trans(img, create_skew_kernel(45,45))
# newimg = lin_trans(img, func[3]())
# for i in range(2):
#     if i==0:
#         rw = random(0.9, 1.2)
#         rh = random(0.9, 1.2)
#         newimg = lin_trans(img, create_scale_kernel(rw, rh))
#         cv2.imshow('scaled, rw = ' + str(rw) + ', rh = ' + str(rh), newimg)
#     elif i==1:
#         ax = random(-15, 15)
#         ay = random(-15, 15)
#         newimg = lin_trans(img, create_skew_kernel(ax, ay))
#         cv2.imshow('skewed, ax = ' + str(ax) + ', ay = ' + str(ay), newimg)
#     elif i==2:
#         dx = random(-20, 20)
#         dy = random(-20, 20)
#         newimg = lin_trans(img, create_offset_kernel(dx, dy))
#         cv2.imshow('offseted, dx = ' + str(dx) + ', dy = ' + str(dy), newimg)
#     elif i==3:
#         newimg = lin_trans(img, horizontal_flip_kernel())
#         cv2.imshow('h_flipped', newimg)
#     elif i==4:
#         newimg = lin_trans(img, vertical_flip_kernel())
#         cv2.imshow('v-flipped', newimg)
#     elif i==5:
#         a = rd.choice([10, 20, -10, -20, 90, -90, 45, -45])
#         newimg = lin_trans(img, create_rotate_kernel(a))
#         cv2.imshow('rotated, a = ' + str(a), newimg)






# print(create_skew_kernel(0,-30))

cv2.waitKey(0);
cv2.destroyAllWindows();



