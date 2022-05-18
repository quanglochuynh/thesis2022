from statistics import median
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import skew, kurtosis, mode, median_absolute_deviation

# gm = 0.75
# dx = 10

def linear_fn(x, c):
    if (x+c>255):
        return 255
    return x+c

def curved(x, c):
    return 255*np.power(x/255, 1/c)


def init_lut(fn, coefficient):
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
    histogram = np.zeros(shape=(3,256), dtype=np.int32)
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            for k in range(im_shape[2]):
                val = image[i][j][k]
                if val>8:
                    histogram[k][val] = histogram[k][val] + 1
    if plot==True:
        plt.rcParams["figure.figsize"] = (6,6)
        ax = plt.gca()
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 2500])
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),histogram[0], 'r')
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),histogram[1], 'g')
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),histogram[2], 'b')
    return histogram

def statistic_extractor(image):
    im = np.asarray(image, dtype=np.float32)
    # create histogram
    shape = np.shape(im)
    total_pix = shape[0]*shape[1]
    (sumR, sumG, sumB) = (np.sum(im[:, :, 0]), np.sum(im[:, :, 1]), np.sum(im[:, :, 2]))
    meanR, meanG, meanB = sumR/total_pix, sumG/total_pix, sumB/total_pix

    modeR, modeG, modeB = mode(im[:, :, 0].flatten(), axis=None), mode(im[:, :, 1].flatten(), axis=None), mode(im[:, :, 2].flatten(), axis=None)
    medianR, medianG, medianB = median_absolute_deviation(im[:, :, 0].flatten(), axis=None), median_absolute_deviation(im[:, :, 1].flatten(), axis=None), median_absolute_deviation(im[:, :, 2].flatten(), axis=None)

    stdR = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 0],meanR)))/(total_pix-1))
    stdG = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 1],meanG)))/(total_pix-1))
    stdB = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 2],meanB)))/(total_pix-1))
    skewR, skewG, skewB = skew(im[:, :, 0].flatten()), skew(im[:, :, 1].flatten()), skew(im[:, :, 2].flatten())
    kurtosisR, kurtosisG, kurtosisB = kurtosis(im[:, :, 0].flatten()), kurtosis(im[:, :, 1].flatten()), kurtosis(im[:, :, 2].flatten())
    return np.array([
        [meanR, meanG, meanB],
        [modeR, modeG, modeB],
        [medianR, medianG, medianB],
        [stdR, stdG, stdB],
        [skewR, skewG, skewB],
        [kurtosisR, kurtosisG, kurtosisB]
    ], dtype=object)

def geometry_extractor(image):
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
            convexHull = cv2.convexHull(cnt)
            convex_area = cv2.contourArea(convexHull)
            # cv2.drawContours(image, [convexHull], -1, (255, 0, 0), 2)

            perimeter = cv2.arcLength(cnt,True)
            # approximatedShape = cv2.approxPolyDP(cnt, 0.002 * perimeter, True)
            # print(perimeter, approximatedShape)
            # cv2.drawContours(image, [approximatedShape], -1, (255, 255, 0), 2)

            (centerXCoordinate, centerYCoordinate), eq_radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(image, (int(centerXCoordinate), int(centerYCoordinate)), int(eq_radius), (0,0,255), 2)
            cv2.circle(image, (int(centerXCoordinate), int(centerYCoordinate)), 10, (255,0,0), 5)

            ellipse = cv2.fitEllipse(convexHull)
            (eX, eY), (alX, alY), orientation = ellipse
            foci_distance = np.sqrt((alX/2)**2 + (alY/2)**2)
            ellipse_eccentricity = max(alX, alY)/foci_distance
            cv2.ellipse(image, center=(int(eX),int(eY)), axes=(int(alX/2),int(alY/2)), angle=int(orientation), startAngle=0, endAngle=360, color=(255, 255, 255), thickness=2)

            moment = cv2.moments(cnt)
            real_area = moment['m00']
            centroidXCoordinate = int(moment['m10'] / real_area)
            centroidYCoordinate = int(moment['m01'] / real_area)
            cv2.circle(image, (int(centroidXCoordinate), int(centroidYCoordinate)), 10, (0,255,0), 5)

            solidity = real_area/convex_area
            bb_ratio = real_area/(w*h)
            eccentricity_distance = np.sqrt( (centerXCoordinate-centroidXCoordinate)**2 + (centerYCoordinate-centroidYCoordinate)**2 )
            return real_area, perimeter, (alX, alY), orientation, ellipse_eccentricity, convex_area, eq_radius, solidity, bb_ratio, eccentricity_distance


def draw_bb(im,data):
    res = np.copy(im)
    (x,y,w,h) = data
    return cv2.rectangle(res,(x,y),(x+w,y+h),(200,0,0),2)