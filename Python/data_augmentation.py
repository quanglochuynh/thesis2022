import cv2
import numpy as np
from numpy import random as rd
import multiprocessing

im_wid_input = 512
im_hei_input = 512
im_wid_output = 224
im_hei_output = 224

input_dim = (im_wid_input,im_hei_input)
output_dim = (im_wid_output,im_hei_output)


gm = 0.75

def random(a,b):
    return np.round(a + rd.random()*(b-a), 2)

def create_scale_kernel(dw,dh):
    tx = -im_wid_input*((dw-1)/2)
    ty = -im_hei_input*((dh-1)/2)
    return np.float32([[dw, 0, tx],
                       [0, dh, ty]])

def create_skew_kernel(ax, ay):
    mx = im_wid_input/2
    my = im_hei_input/2
    kernel = np.float32([[1, ax,  -ax*mx/2],
                        [ay, 1, -ay*my/2]])
    return kernel

def create_offset_kernel(dx, dy):
    return np.float32([[1, 0, dx], [0, 1, dy]])

def horizontal_flip_kernel():
    return np.float32([[-1, 0, im_wid_input], [0, 1, 0]])

def vertical_flip_kernel():
    return np.float32([[1, 0, 0], [0, -1, im_hei_input]])

def create_rotate_kernel(angle):
    a = angle * 2*np.pi/360
    mx = im_wid_input/2
    my = im_hei_input/2
    alp = np.cos(a)
    bet = np.sin(a)
    kernel = np.float32([[alp, bet, (1-alp)*mx - bet*my],
                        [-bet, alp, bet*mx + (1-alp)*my]])
    return kernel

def lin_trans(source, kernel):
    return cv2.warpAffine(source, kernel, (source.shape[0], source.shape[1]))

def augment(source, address):
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
                a = rd.choice([10, 10, 20, 20, -10, -10, -20, -20, 90, -90, 45, -45])
                newimg = lin_trans(newimg, create_rotate_kernel(a))
    newimg = cv2.resize(newimg, output_dim)
    cv2.imwrite(address, newimg)
    print('img saved to: ' + address)


classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']
inp_address = 'D:/Thesis_data/Color_Corrected_512x512/'
training_address = 'D:/Thesis_data/training_img/'
testing_address = 'D:/Thesis_data/testing_img/'
class_id = 6
img_id = 9


def batch_augment(class_id):
    n = 1
    m = 1
    for img_id in range(1,81):
        img = cv2.imread(inp_address + classes_name[class_id] + '/image (' + str(img_id) + ').JPG')
        if img_id<80:
            cv2.imwrite(training_address + classes_name[class_id] + '/image(' + str(n) + ').JPG', cv2.resize(img,output_dim))
            for k in range(11):
                n = n+1
                augment(img, training_address + classes_name[class_id] + '/image(' + str(n) + ').JPG')
            n = n+1
        else:
            cv2.imwrite(testing_address + classes_name[class_id] + '/image(' + str(m) + ').JPG', cv2.resize(img,output_dim))
            for k in range(11):
                m = m+1
                augment(img, testing_address + classes_name[class_id] + '/image(' + str(m) + ').JPG')
            m = m+1
            

if __name__ == '__main__':
    k = 0
    p1 = multiprocessing.Process(target=batch_augment, args=(k+0,))
    p2 = multiprocessing.Process(target=batch_augment, args=(k+1,))
    p3 = multiprocessing.Process(target=batch_augment, args=(k+2,))
    p4 = multiprocessing.Process(target=batch_augment, args=(k+3,))
    p5 = multiprocessing.Process(target=batch_augment, args=(k+4,))
    p6 = multiprocessing.Process(target=batch_augment, args=(k+5,))
    p7 = multiprocessing.Process(target=batch_augment, args=(k+6,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()

    p1.join()
    p2.join()
    p3.join()  
    p4.join()
    p5.join()
    p6.join()
    p7.join()

print("Done!")

