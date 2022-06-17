import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

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
        ax = plt.gca()
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 2500])
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),histogram[0], 'r')
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),histogram[1], 'g')
        plt.plot(np.linspace(0,255, 256,dtype=np.int16),histogram[2], 'b')
        plt.show()
    return histogram

def statistic_analysis(image):
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
    ], dtype=np.float32).flatten()

def geometry_analysis(cnt, ellipse):
        x,y,w,h = cv2.boundingRect(cnt)
        convexHull = cv2.convexHull(cnt)
        convex_area = cv2.contourArea(convexHull)
        perimeter = cv2.arcLength(cnt,True)
        (centerXCoordinate, centerYCoordinate), eq_radius = cv2.minEnclosingCircle(cnt)
        (eX, eY), (alX, alY), orientation = ellipse
        foci_distance = np.sqrt((alX/2)**2 + (alY/2)**2)
        ellipse_eccentricity = max(alX, alY)/foci_distance
        moment = cv2.moments(cnt)
        real_area = moment['m00']
        if (real_area<1):return [0]
        centroidXCoordinate = int(moment['m10'] / real_area)
        centroidYCoordinate = int(moment['m01'] / real_area)
        solidity = real_area/convex_area
        bb_ratio = real_area/(w*h)
        eccentricity_distance = np.sqrt( (centerXCoordinate-centroidXCoordinate)**2 + (centerYCoordinate-centroidYCoordinate)**2 )
        return [w,h,real_area, perimeter, alX, alY, ellipse_eccentricity, convex_area, eq_radius, solidity, bb_ratio, eccentricity_distance]


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

def preprocess_hsv(image_bgr, lut1=None, lut2=None, Contour=True, origin_bgr=False):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if (lut2!=None):
        image_hsv = apply_lut(image_hsv, 2, lut2)   #tang brightness
    if (lut1!=None):
        image_hsv = apply_lut(image_hsv, 1, lut1)   #tang Sat
    image_hsv = hsv_filter(image_hsv)
    x,y,w,h,cnt = bounding_box(image_hsv[:,:,2])
    ellipse = cv2.fitEllipse(cnt)
    (eX, eY), (alX, alY), orientation = ellipse
    if orientation>90:
        orientation = -180+orientation
    a = orientation * 2*np.pi/360
    im_wid_input, im_hei_input = np.shape(image_bgr[:,:,2])
    mx = im_wid_input/2
    my = im_hei_input/2
    alp = np.cos(a)
    bet = np.sin(a)
    image_hsv = cv2.warpAffine(image_hsv, np.float32([[alp, bet, (1-alp)*mx - bet*my], [-bet, alp, bet*mx + (1-alp)*my]]), (im_wid_input, im_hei_input))
    if origin_bgr:
        rot_bgr = cv2.warpAffine(image_bgr, np.float32([[alp, bet, (1-alp)*mx - bet*my], [-bet, alp, bet*mx + (1-alp)*my]]), (im_wid_input, im_hei_input))
    x,y,w,h,cnt2 = bounding_box(rot_bgr[:,:,0])
    if Contour==True:
        if origin_bgr:
            return aspect_crop(image_hsv, x,y,w,h), cnt2, ellipse, rot_bgr
        return aspect_crop(image_hsv, x,y,w,h), cnt2, ellipse                    # return ellipse
    else:
        return aspect_crop(image_hsv, x,y,w,h)

class feature_extract:
    def __init__(self) -> None:
        self.lut1 = init_lut(fn=linear_fn, coefficient=5)
        self.lut2 = init_lut(fn=curved, coefficient=1.5)
        self.image_hsv = None
        self.image_rgb = None
        self.origin_rgb = None
        self.clahe6 = cv2.createCLAHE(6, (8,8))
        self.clahe1 = cv2.createCLAHE(1, (8,8))
        self.clahe4 = cv2.createCLAHE(4, (8,8))
        self.im_size = (128,256)
        self.gabor_filter = []
        theta = [0, np.pi/6,  np.pi/3, np.pi/2, -np.pi/3, -np.pi/6]
        for i in range(len(theta)):
            self.gabor_filter.append(cv2.getGaborKernel((35,35), sigma=6, theta=theta[i], lambd=6*np.pi, gamma=0.2, psi=0))
        self.cnt = None
        self.ellipse = None
        self.clahe_v = None
        self.level = 16
        self.overall_geometry = None
        self.overall_rgb_stat = None
        self.overall_hsv_stat = None
        self.grid_stat = None
        self.glcm_grid = None
        self.n1 = None
        self.n2 = None
        self.structure = None
        self.mold = None
        self.bins = np.linspace(0, 256, self.level+1)
        self.glcm_dissimilarity = None
        self.glcm_correlation = None
        pass

    def extract(self, image_bgr):
        self.pre_process(image_bgr)
        self.overall_geometry = geometry_analysis(self.cnt, self.ellipse)
        self.overall_rgb_stat = statistic_analysis(self.origin_rgb)
        self.overall_hsv_stat = statistic_analysis(self.image_hsv)
        self.extract_structure()
        self.extract_mold()
        self.extract_glcm()
        self.extract_grid()

    def pre_process(self, image_bgr):
        self.image_hsv, self.cnt, self.ellipse, image2 = preprocess_hsv(image_bgr, self.lut1, self.lut2, Contour=True, origin_bgr=True)
        self.image_hsv = cv2.resize(self.image_hsv, self.im_size)
        self.image_rgb = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2RGB)
        x,y,w,h = cv2.boundingRect(self.cnt)
        # self.origin_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        self.origin_rgb = cv2.resize(cv2.cvtColor(image2[y:y+h, x:x+w], cv2.COLOR_BGR2RGB),self.im_size)
        self.clahe_v = self.clahe6.apply(self.image_hsv[:,:,2])

        
    def extract_structure(self):    
        gabored = []
        for i in range(len(self.gabor_filter)):
            gabored.append(cv2.filter2D(self.clahe_v, cv2.CV_8UC3, -self.gabor_filter[i]))

        bbrect_array = []
        for i in range(len(self.gabor_filter)):
            contours, hir = cv2.findContours(gabored[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            hei,wid = np.shape(self.clahe_v)
            bbrect = []
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if x<20:
                    continue
                if (x+w)>(wid-20):
                    continue
                if y<20:
                    continue
                if (y+h)>(hei-20):
                    continue
                convexHull = cv2.convexHull(cnt)
                convexhull_area = cv2.contourArea(convexHull)
                if convexhull_area<16:
                    continue
                if len(cnt)<5:
                        continue
                if len(convexHull)<5:
                        continue
                rect = cv2.minAreaRect(convexHull)
                (x,y), (w,h), o = rect
                bbrect.append([x,y,w,h])
            if len(bbrect) == 0:
                bbrect_array.append([[0,0,0,0]])
            else:
                bbrect_array.append(bbrect)  
        self.n1 =[]
        mean = [] 
        std = []
        skewness = []
        kurtosises = []
        for i in range(len(bbrect_array)):
            self.n1.append(np.shape(bbrect_array[i])[0])
            mean.append(np.mean(bbrect_array[i], axis=0).tolist())
            std.append(np.std(bbrect_array[i], axis=0).tolist())
            skewness.append(skew(bbrect_array[i], axis=0).tolist())
            kurtosises.append((-kurtosis(bbrect_array[i], axis=0)).tolist())
        self.structure = np.asarray([mean, std, skewness, kurtosises]).flatten()
        
    def extract_mold(self):
        r,g,b = cv2.split(self.image_rgb)

        ret, r = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
        ret, g = cv2.threshold(g, 135, 255, cv2.THRESH_BINARY)
        ret, b = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)

        meg = np.min([r,g,b], axis=0)
        meg = cv2.GaussianBlur(meg, (11,11),0)
        ret, meg = cv2.threshold(meg, 20, 255, cv2.THRESH_BINARY)

        contours, hir = cv2.findContours(meg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hei,wid = np.shape(meg)
        meg = cv2.cvtColor(meg, cv2.COLOR_GRAY2RGB)
        shape = []
        stat = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            convexHull = cv2.convexHull(cnt)
            convexhull_area = cv2.contourArea(convexHull)
            if convexhull_area < 9:
                continue
            if len(cnt)<5:
                continue
            if len(convexHull)<5:
                continue
            (x2,y2),(w2,h2),o = cv2.minAreaRect(convexHull)
            r = w2/h2
            if (r>3) or (r<0.333):
                continue
            stat.append(statistic_analysis(self.image_rgb[y:y+h, x:x+w, :]))
            shape.append([w,h])

        self.n2 = np.shape(stat)[0]
        if (self.n2 != 0):
            ft = np.concatenate([shape, stat], axis=1).tolist()
            mean = np.mean(ft, axis=0).tolist()
            stddev = np.std(ft,axis=0).tolist()
            skewness = skew(ft,axis=0)
            kurtosises = kurtosis(ft,axis=0)
            self.mold = np.asarray([mean, stddev, skewness, kurtosises]).flatten()
        else:
            self.mold = np.zeros(56)
        
    def extract_glcm(self):
        self.clahe_v = self.clahe4.apply(self.image_hsv[:,:,2])
        digitize = np.digitize(self.clahe_v, self.bins) - 1
        glcm = graycomatrix(digitize, [5,7,9], [0, np.pi/4, np.pi/2, 3*np.pi/4], self.level, True, False)
        glcm = glcm[2:self.level+1,2:self.level+1]
        self.glcm_dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        self.glcm_correlation = graycoprops(glcm, 'correlation').flatten()

    def extract_glcm_grid(self):
        self.glcm_grid = []
        hei, wid, c = np.shape(self.origin_rgb)
        h,w = int(hei/4), int(wid/4)
        self.clahe_v = self.clahe4.apply(self.image_hsv[:,:,2])
        digitize = np.digitize(self.clahe_v, self.bins) - 1
        for i in range(0, 4):
            for j in range(0, 4):
                glcm = graycomatrix(digitize[i*h:i*h+int(hei/4), j*w:j*w+int(wid/4)], [7,9], [0, np.pi/4, np.pi/2, 3*np.pi/4], self.level, True, False)
                self.glcm_grid = np.concatenate([self.glcm_grid, graycoprops(glcm, 'dissimilarity').flatten(), graycoprops(glcm, 'correlation').flatten()], axis=None)

    def extract_color_grid(self):
        self.grid_stat = []
        hei, wid, c = np.shape(self.origin_rgb)
        h,w = int(hei/8), int(wid/4)
        for i in range(1, 7):
            for j in range(0, 4):
                self.grid_stat = np.concatenate([self.grid_stat, statistic_analysis(self.origin_rgb[i*h:i*h+int(hei/6), j*w:j*w+int(wid/4),:])], axis=None)
                
    def extract_compress_HSV(self):
        
        pass