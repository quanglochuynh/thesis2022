import cv2
import numpy as np
from numpy import random as rd, uint8
import multiprocessing

im_wid = 512
im_hei = 512
gm = 0.75


def curved(x):
    if x < 10:
        return 0
    else:
        return 255*np.power(x/255, 1/gm)

def image_correct(img, channel):
    newimg = np.zeros(np.shape(img), dtype=uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newimg[i][j] = img[i][j];
            newimg[i][j][channel] = curved(img[i][j][channel])
    return newimg

classes_name = ['Agglutinated', 'Brittle', 'Compartmentalized_Brown', 'Compartmentalized_PartiallyPurple', 'Compartmentalized_Purple', 'Compartmentalized_Slaty', 'Compartmentalized_White', 'Flattened', 'Moldered', 'Plated_Brown', 'Plated_PartiallyPurple', 'Plated_Purple', 'Plated_Slaty', 'Plated_White']
inp_address = 'D:/Thesis_data/Ver4_MedB/'
out_address = 'D:/Thesis_data/Color_Corrected/'

def batch_correct(class_id):
    for img_id in range(1,101):
        img = cv2.imread(inp_address + classes_name[class_id] + '/image (' + str(img_id) + ').JPG')
        img = cv2.resize(img, (im_wid,im_hei))
        img = image_correct(img, 1)
        cv2.imwrite(out_address + classes_name[class_id] + '/image (' + str(img_id) + ').JPG', cv2.resize(img,(256,256)))
        print("Done, "+ classes_name[class_id] +" image "+ str(img_id))

# batch_correct(0)
n = 12
if __name__ == '__main__':
    p1 = multiprocessing.Process(target=batch_correct, args=(n+0,))
    p2 = multiprocessing.Process(target=batch_correct, args=(n+1,))
    # p3 = multiprocessing.Process(target=batch_correct, args=(n+2,))
    # p4 = multiprocessing.Process(target=batch_correct, args=(n+3,))
    # p5 = multiprocessing.Process(target=batch_correct, args=(n+4,))
    # p6 = multiprocessing.Process(target=batch_correct, args=(n+5,))

    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()

    p1.join()
    p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
