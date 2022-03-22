import cv2
import numpy as np
from numpy import random as rd, uint8
import multiprocessing


im_wid = 512
im_hei = 512
gm = 0.75


def curved(x):
    return 255*np.power(x/255, 1/gm)

def image_correct(img, channel):
    newimg = np.zeros(np.shape(img), dtype=uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newimg[i][j] = img[i][j];
            newimg[i][j][channel] = curved(img[i][j][channel])
    return newimg

def gammaCurve(frame, gm):
    look_up_table = np.zeros((256,3), dtype = 'uint8')
    for i in range(256):
        look_up_table[i][0] = pow(i / 255, 1 / gm) * 255
        look_up_table[i][1] = i
        look_up_table[i][2] = i
    return cv2.LUT(frame, look_up_table)


def random(a,b):
    return np.round(a + rd.random()*(b-a), 2)

def create_scale_kernel(dw,dh):
    tx = -im_wid*((dw-1)/2)
    ty = -im_hei*((dh-1)/2)
    return np.float32([[dw, 0, tx],
                       [0, dh, ty]])

def create_skew_kernel(ax, ay):
    mx = im_wid/2
    my = im_hei/2
    kernel = np.float32([[1, ax,  -ax*mx/2],
                        [ay, 1, -ay*my/2]])
    return kernel

def create_offset_kernel(dx, dy):
    return np.float32([[1, 0, dx], [0, 1, dy]])

def horizontal_flip_kernel():
    return np.float32([[-1, 0, im_wid], [0, 1, 0]])

def vertical_flip_kernel():
    return np.float32([[1, 0, 0], [0, -1, im_hei]])

def create_rotate_kernel(angle):
    a = angle * 2*np.pi/360
    mx = im_wid/2
    my = im_hei/2
    alp = np.cos(a)
    bet = np.sin(a)
    kernel = np.float32([[alp, bet, (1-alp)*mx - bet*my],
                        [-bet, alp, bet*mx + (1-alp)*my]])
    return kernel

def lin_trans(source, kernel):
    return cv2.warpAffine(source, kernel, (source.shape[0], source.shape[1]))

def augment(source):
    newimg = source;
    for i in range(6):
        if (rd.rand()<0.4):
            if i==0:
                rw = random(0.9, 1.2)
                rh = random(0.9, 1.15)
                newimg = lin_trans(newimg, create_scale_kernel(rw, rh))
            elif i==1:
                ax = random(-0.2, 0.2)
                ay = random(-0.2, 0.2)
                newimg = lin_trans(newimg, create_skew_kernel(ax, ay))
            elif i==2:
                dx = random(-20, 20)
                dy = random(-20, 20)
                newimg = lin_trans(newimg, create_offset_kernel(dx, dy))
            elif i==3:
                newimg = lin_trans(newimg, horizontal_flip_kernel())
            elif i==4:
                newimg = lin_trans(newimg, vertical_flip_kernel())
            elif i==5:
                a = rd.choice([10, 20, -10, -20, 90, -90, 45, -45])
                newimg = lin_trans(newimg, create_rotate_kernel(a))
    return cv2.resize(newimg, (256,256))

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']
inp_address = 'D:/Thesis_data/Ver4_MedB/'
img_id = 9
#LINUX
# img = cv2.imread('/home/flint/Documents/thesis2022/Python/data/bean.JPG')
#MacOS
# img = cv2.imread('/Users/lochuynhquang/Documents/thesis2022/Python/data/bean.JPG')
#Windows
# img = cv2.imread('C:/Users/quang/Documents/thesis2022/Python/data/bean.JPG')
img = cv2.imread(inp_address + classes_name[6] + '/image (' + img_id + ').JPG')
img = cv2.resize(img, (im_wid,im_hei))

img = image_correct(img, 1)
cv2.imshow('original', img)

# for j in range(10):
#     newimg = augment(img)
#     # cv2.imshow('augmented' + str(j), newimg)
#     cv2.imwrite(address + classes_name[6] + 'image (' + str(j) + ').JPG',newimg)




cv2.waitKey(0);
cv2.destroyAllWindows();



