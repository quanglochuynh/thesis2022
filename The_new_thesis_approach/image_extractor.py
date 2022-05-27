import cv2
from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import idst 
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
import statistics as st

# gm = 0.75
# dx = 10

def partition(l, r, nums):
    # Last element will be the pivot and the first element the pointer
    pivot, ptr = nums[r][1], l
    for i in range(l, r):
        if nums[i][1] >= pivot:
            # Swapping values smaller than the pivot to the front
            nums[i], nums[ptr] = nums[ptr], nums[i]
            ptr += 1
    # Finally swappping the last element with the pointer indexed number
    nums[ptr], nums[r] = nums[r], nums[ptr]
    return ptr

def quicksort(l, r, nums):
    if len(nums) == 1:  # Terminating Condition for recursion. VERY IMPORTANT!
        return nums
    if l < r:
        pi = partition(l, r, nums)
        quicksort(l, pi-1, nums)  # Recursively sorting the left values
        quicksort(pi+1, r, nums)  # Recursively sorting the right values
    return nums

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
            if image[i][j][2]<40:
                image[i][j] = np.array([0, 0, 0])
    # h,s,v = cv2.split(im)
    # th_v = cv2.threshold(v, 25, 255, cv2.THRESH_TOZERO)

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

def statistic_analysis(image):
    im = np.asarray(image, dtype=np.float32)
    # create histogram
    shape = np.shape(im)
    total_pix = shape[0]*shape[1]
    (sumR, sumG, sumB) = (np.sum(im[:, :, 0]), np.sum(im[:, :, 1]), np.sum(im[:, :, 2]))
    meanR, meanG, meanB = sumR/total_pix, sumG/total_pix, sumB/total_pix
    medianR, medianG, medianB = st.median(im[:, :, 0].flatten()), st.median(im[:, :, 1].flatten()), st.median(im[:, :, 2].flatten())
    stdR = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 0],meanR)))/(total_pix-1))
    stdG = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 1],meanG)))/(total_pix-1))
    stdB = np.sqrt(np.sum(np.square(np.subtract(im[:, :, 2],meanB)))/(total_pix-1))
    skewR, skewG, skewB = skew(im[:, :, 0].flatten()), skew(im[:, :, 1].flatten()), skew(im[:, :, 2].flatten())
    kurtosisR, kurtosisG, kurtosisB = kurtosis(im[:, :, 0].flatten()), kurtosis(im[:, :, 1].flatten()), kurtosis(im[:, :, 2].flatten())
    return np.array([
        [meanR, meanG, meanB],
        [medianR, medianG, medianB],
        [stdR, stdG, stdB],
        [skewR, skewG, skewB],
        [kurtosisR, kurtosisG, kurtosisB]
    ], dtype=object)

def geometry_analysis(cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        convexHull = cv2.convexHull(cnt)
        convex_area = cv2.contourArea(convexHull)
        # cv2.drawContours(image, [convexHull], -1, (255, 0, 0), 2)
        perimeter = cv2.arcLength(cnt,True)
        # approximatedShape = cv2.approxPolyDP(cnt, 0.002 * perimeter, True)
        # print(perimeter, approximatedShape)
        # cv2.drawContours(image, [approximatedShape], -1, (255, 255, 0), 2)
        (centerXCoordinate, centerYCoordinate), eq_radius = cv2.minEnclosingCircle(cnt)
        # cv2.circle(image, (int(centerXCoordinate), int(centerYCoordinate)), int(eq_radius), (0,0,255), 2)
        # cv2.circle(image, (int(centerXCoordinate), int(centerYCoordinate)), 10, (255,0,0), 5)
        # print('CVH', len(convexHull))
        if (len(convexHull)>4):
            ellipse = cv2.fitEllipse(convexHull)
            (eX, eY), (alX, alY), orientation = ellipse
            foci_distance = np.sqrt((alX/2)**2 + (alY/2)**2)
            ellipse_eccentricity = max(alX, alY)/foci_distance
            # cv2.ellipse(image, center=(int(eX),int(eY)), axes=(int(alX/2),int(alY/2)), angle=int(orientation), startAngle=0, endAngle=360, color=(255, 255, 255), thickness=2)

            moment = cv2.moments(cnt)
            real_area = moment['m00']
            if (real_area<1):return [0]
            centroidXCoordinate = int(moment['m10'] / real_area)
            centroidYCoordinate = int(moment['m01'] / real_area)
            # cv2.circle(image, (int(centroidXCoordinate), int(centroidYCoordinate)), 10, (0,255,0), 5)

            solidity = real_area/convex_area
            bb_ratio = real_area/(w*h)
            eccentricity_distance = np.sqrt( (centerXCoordinate-centroidXCoordinate)**2 + (centerYCoordinate-centroidYCoordinate)**2 )
            return x,y,w,h,real_area, perimeter, alX, alY, orientation, ellipse_eccentricity, convex_area, eq_radius, solidity, bb_ratio, eccentricity_distance
        else:
            # print('CLGT!!!!!!!!!!!!')
            return [0]

def draw_bb(im,data):
    res = np.copy(im)
    (x,y,w,h) = data
    return cv2.rectangle(res,(x,y),(x+w,y+h),(200,0,0),2)

def HE(grey_img, grey=False):
    if grey==False:
        grey_img = cv2.cvtColor(grey_img, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(grey_img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[grey_img]

def bounding_box(image):
    ret, im = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    contours, hie = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>50) and (w>100):
            return x,y,w,h,cnt
    return 0,0,0,0,[]

def seperate_chanel(image, plot=False):
    h,s,v = image[:,:,0], image[:,:,1], image[:,:,2]
    if plot==True:
        plt.subplot(1,3,1)
        plt.imshow(h)
        plt.subplot(1,3,2)
        plt.imshow(s)
        plt.subplot(1,3,3)
        plt.imshow(v)
        plt.show()
    return h,s,v


def hsv_contour_extract(image_hsv):
    h,s,v = seperate_chanel(image_hsv, plot=False)
    # v = CLAHE(v, grey=True)
    clahe_op = cv2.createCLAHE(4, (8,8))
    v = clahe_op.apply(v)
    ret, thresh_h = cv2.threshold(h, 120,255,cv2.THRESH_TOZERO)
    # ret, thresh_h = cv2.threshold(h, 200,255,cv2.THRESH_TOZERO_INV)
    ret, thresh_v = cv2.threshold(v, 30,255,cv2.THRESH_TOZERO_INV)
    ret, thresh_v2 = cv2.threshold(v, 192,255,cv2.THRESH_TOZERO)

    ct_H = []
    ct_V = []
    contour_H ,h = cv2.findContours(thresh_h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour_H:
        convexHull = cv2.convexHull(cnt)
        convexhull_area = cv2.contourArea(convexHull)
        if (convexhull_area>100) and (convexhull_area<400000):
            perimeter = cv2.arcLength(cnt,True)
            approximatedShape = cv2.approxPolyDP(cnt, 0.004 * perimeter, True)
            # cv2.drawContours(origin_rgb, [approximatedShape], -1, (255, 255, 0), 2)
            ct_H.append(approximatedShape)

    contour_V ,h = cv2.findContours(thresh_v, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour_V:
        convexHull = cv2.convexHull(cnt)
        convexhull_area = cv2.contourArea(convexHull)
        if (convexhull_area>100) and (convexhull_area<400000):
            perimeter = cv2.arcLength(cnt,True)
            approximatedShape = cv2.approxPolyDP(cnt, 0.001 * perimeter, True)
            # cv2.drawContours(origin_rgb, [approximatedShape], -1, (0, 255, 255), 1)
            ct_V.append(approximatedShape)

    contour_V ,h = cv2.findContours(thresh_v2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour_V:
        convexHull = cv2.convexHull(cnt)
        convexhull_area = cv2.contourArea(convexHull)
        if (convexhull_area>100) and (convexhull_area<200000):
            perimeter = cv2.arcLength(cnt,True)
            approximatedShape = cv2.approxPolyDP(cnt, 0.001 * perimeter, True)
            # cv2.drawContours(origin_rgb, [approximatedShape], -1, (0, 255, 255), 1)
            ct_V.append(approximatedShape)

    return(ct_H, ct_V)

def aspect_crop(image, x,y,w,h):
    c = int((w-max(int(h/2),w))/2)
    return image[y:y+h,x+c:x+max(w,int(h/2))+c]

def preprocess_hsv(image):
    lut1 = init_lut(fn=linear_fn, coefficient=15)
    lut2 = init_lut(fn=curved, coefficient=1.5)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = apply_lut(image_hsv, 2, lut2)   #tang brightness
    image_hsv = apply_lut(image_hsv, 1, lut1)   #tang Sat
    image_hsv = hsv_filter(image_hsv)
    x,y,w,h,cnt = bounding_box(image_hsv[:,:,2])
    ellipse = cv2.fitEllipse(cnt)
    (eX, eY), (alX, alY), orientation = ellipse
    print('ori = ', orientation)
    if orientation>90:
        orientation = -180+orientation
    a = orientation * 2*np.pi/360
    im_wid_input, im_hei_input = np.shape(image[:,:,2])
    mx = im_wid_input/2
    my = im_hei_input/2
    alp = np.cos(a)
    bet = np.sin(a)
    kernel = np.float32([[alp, bet, (1-alp)*mx - bet*my],
                        [-bet, alp, bet*mx + (1-alp)*my]])
    image_hsv = cv2.warpAffine(image_hsv, kernel, (im_wid_input, im_hei_input))
    x,y,w,h,cnt = bounding_box(image_hsv[:,:,2])
    image_hsv_croped = aspect_crop(image_hsv, x,y,w,h)
    return image_hsv_croped, cnt

def select_feature(image):
    image_hsv, out_cnt = preprocess_hsv(image)
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    # x,y,w,h,out_cnt = bounding_box(image_hsv[:,:,2])

    overall_geometry = geometry_analysis(out_cnt)
    overall_statistic = statistic_analysis(image_rgb[y:y+h,x:x+w])

    # image_hsv_croped = aspect_crop(image_hsv, x,y,w,h)
    hue,sat,val = seperate_chanel(image_hsv, plot=False)

    geometry_feature = []
    statistic_feature = []
    selected_geometry_feature = [overall_geometry]
    selected_statistic_feature = [overall_statistic]
    ct_H, ct_V = hsv_contour_extract(image_hsv)
    area = []
    id = 0
    for cnt in ct_H:
        (x,y,w,h) = cv2.boundingRect(cnt)
        geo = geometry_analysis(cnt)
        # print(geo)
        if len(geo)>1:
            area.append((id,w*h))
            id = id+1
            geometry_feature.append(geo)
            statistic_feature.append(statistic_analysis(image_rgb[y:y+h,x:x+w]))

    for cnt in ct_V:
        (x,y,w,h) = cv2.boundingRect(cnt)
        geo = geometry_analysis(cnt)
        # print(geo)
        if len(geo)>1:
            area.append((id,w*h))
            id = id+1
            geometry_feature.append(geo)
            statistic_feature.append(statistic_analysis(image_rgb[y:y+h,x:x+w]))

    area = quicksort(0, len(area)-1, area)
    # print('LEN=',len(geometry_feature))
    if (len(geometry_feature)>10):
        for i in range(10):
            # print(area[i][0], area[i][1])
            selected_geometry_feature.append(geometry_feature[area[i][0]])
            selected_statistic_feature.append(statistic_feature[area[i][0]])
    else:
        for i in range(10):
            id = np.random.randint(0, len(area))
            selected_geometry_feature.append(geometry_feature[id])
            selected_statistic_feature.append(statistic_feature[id])

    # val = CLAHE(val, grey=True)
    clahe_op = cv2.createCLAHE(4, (8,8))
    val = clahe_op.apply(val)
    val = cv2.resize(val, (128,256))

    bins = np.linspace(0, 256, 33)
    digitize = np.digitize(val, bins) - 1

    glcm = graycomatrix(digitize, [5,7,9,11], [0, np.pi/4, np.pi/2, 3*np.pi/4], 32, True, False)
    glcm = glcm[1:31,1:31,:,:];
    glcm_cons = graycoprops(glcm, 'contrast')
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')
    glcm_energy = graycoprops(glcm, 'energy')
    glcm_correlation = graycoprops(glcm, 'correlation')

    return np.concatenate([np.array(selected_geometry_feature).flatten(), np.array(selected_statistic_feature).flatten(), glcm_cons.flatten(), glcm_dissimilarity.flatten(), glcm_energy.flatten(), glcm_correlation.flatten()], axis=None)
