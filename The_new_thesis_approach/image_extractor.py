import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import skew, kurtosis
from bezier import bezier_point

# gm = 0.75
# dx = 10

def linear_fn(x, dx):
    if (x+dx>255):
        return 255
    return x+dx

def curved(x, gm):
    return 255*np.power(x/255, 1/gm)
    
def bezier(x1,y1, x2,y2, t):
    x0 = 0
    y0 = 0
    x3 = 255
    y3 = 255
    return bezier_point(x0, x1,x2,x3, t)
    pass

def init_lut(fn=curved, coefficient=0):
    LUT = []
    for i in range(256):
        LUT.append(np.uint8(fn(i, coefficient)))
    return LUT

def apply_lut(img, channel, LUT):

    newimg = np.zeros(np.shape(img), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newimg[i][j] = img[i][j]
            newimg[i][j][channel] = LUT[img[i][j][channel]]
    return newimg

def hsv_filter(im):
    image = im
    sp = np.shape(image)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if image[i][j][2]<20:
                image[i][j] = np.array([0, 0, 0])
    return image 

def histogram_analysis(im, plot=False):
    image = np.asarray(im, dtype=np.int8)
    im_shape = np.shape(im)
    res = np.zeros(shape=(3,256), dtype=np.int32)
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            for k in range(im_shape[2]):
                val = image[i][j][k]
                if val>8:
                    res[k][val] = res[k][val] + 1
    if plot==True:
        plt.rcParams["figure.figsize"] = (6,6)
        ax = plt.gca()
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 2500])
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),res[0], 'r')
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),res[1], 'g')
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),res[2], 'b')
    return res

def statistic_extractor(image):
    im = np.asarray(image, dtype=np.float32)
    # create histogram
    shape = np.shape(im)
    total_pix = shape[0]*shape[1]
    (sumR, sumG, sumB) = (np.sum(im[:, :, 0]), np.sum(im[:, :, 1]), np.sum(im[:, :, 2]))
    meanR, meanG, meanB = sumR/total_pix, sumG/total_pix, sumB/total_pix
    stdR = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 0],meanR)))/(total_pix-1))
    stdG = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 1],meanG)))/(total_pix-1))
    stdB = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 2],meanB)))/(total_pix-1))
    skewR, skewG, skewB = skew(im[:, :, 0].flatten()), skew(im[:, :, 1].flatten()), skew(im[:, :, 2].flatten())
    kurtosisR, kurtosisG, kurtosisB = kurtosis(im[:, :, 0].flatten()), kurtosis(im[:, :, 1].flatten()), kurtosis(im[:, :, 2].flatten())
    return np.array([
        [meanR, meanG, meanB],
        [stdR, stdG, stdB],
        [skewR, skewG, skewB],
        [kurtosisR, kurtosisG, kurtosisB]
    ], dtype=object)

def find_bb(image):
    im = image
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # idx =0 
    for cnt in contours:
        # idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>50) and (h>100):
            # wid = int(max(h,w))
            # im=im[y-max(0,int((w-h)/2)):y+wid-max(0,int((w-h)/2)), x-max(0,int((h-w)/2)):x+wid-max(0,int((h-w)/2))]
            # im = im[y:y+h,x:x+w]
            # break
            return (x,y,w,h)

def draw_bb(im,data):
    res = np.copy(im)
    (x,y,w,h) = data
    return cv2.rectangle(res,(x,y),(x+w,y+h),(200,0,0),2)