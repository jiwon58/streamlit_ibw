from .images import Image
import numpy as np
import imutils
import math
import cv2


class ArrowImage():
    def __init__(self, RefPath, RotPath, angle=90):
        
        self.ref = Image(RefPath)
        self.rot = Image(RotPath)
        
        if self.ref.channels != self.rot.channels:
            print('Two images have different channels.')
        self.init_angle = angle # clock-wise direction
        self.flatten_order = None
    
    def to_img(self, img):
        # uint8 formatting
        return np.uint8(((img+abs(np.amin(img)))/(np.amax(img)+abs(np.amin(img)))*255))
    
    def part_order(self, img, order):
        h, w = img.shape
        h, w = int(h/2), int(w/2)
        if order == 1:
            return img[h:, :w]
        elif order == 2:
            return img[h:, w:]
        elif order == 3:
            return img[:h, :w]
        elif order == 4:
            return img[:h, w:]
        else:
            raise ValueError('Check order.')
            
    def flatten(self, channel, order=-1):
        # If order is -1, no flattening

        self.flattenRef = self.ref.flatten(channel=channel, order=order)
        self.flattenRot = self.rot.flatten(channel=channel, order=order)
        self.flattenOrder = order
        return self.flattenRef, self.flattenRot
    
    def plane_fit(self, channel, order=1):
        # order must be 1 or 2

        self.PFRef = self.ref.doPlaneFit(channel=channel, order=order)
        self.PFRot = self.rot.doPlaneFit(channel=channel, order=order)
        self.PForder = order
        return self.PFRef, self.PFRot

    def calcMatches(self, channel, part, mode, MAX_FEATURE=2000):
        if self.flattenOrder == None or self.flattenOrder == -1:
            self.flatten(channel, order=1)
            print("Flattening is done, order = 1")

        ## Feature position check
        if part:
            img = self.toImg(self.flattenRef)
            flattenRef = self.partorder(img, part)
            flattenRot = imutils.rotate(self.toImg(self.flattenRot), -self.initangle)
            flattenRot = self.partorder(flattenRot, part)
        else:
            flattenRef = self.toImg(self.flattenRef)
            flattenRot = imutils.rotate(self.toImg(self.flattenRot), -self.initangle)
            #flattenRot = flattenRot
        
        if mode == 'SIFT':
            # SIFT is more powerful
            if int(cv2.__version__.split('.')[0]) >= 4:
                model = cv2.SIFT_create()
            else:
                model = cv2.xfeatures2d.SIFT_create()
        elif mode == 'ORB':
            model = cv2.ORB_create(MAX_FEATURE)
        else:
            print('Descriptor model is wrong!')
            return False

        keypoint1, descriptor1 = model.detectAndCompute(flattenRef, None)
        keypoint2, descriptor2 = model.detectAndCompute(flattenRot, None)
        
        # Find match points based on detected descriptions
        matcher = cv2.BFMatcher()
        matches = matcher.match(descriptor1, descriptor2, None)
        # Sort the matches from shortest distance
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        return keypoint1, keypoint2, matches, flattenRef, flattenRot
    
    def drawMatches(self, channel, part=False, GOOD_MATCH_PERCENT = 0.20, mode='SIFT', MAX_FEATURE = 2000):
        keypoint1, keypoint2, matches, Ref, Rot = self.calcMatches(channel, part, mode=mode, MAX_FEATURE=MAX_FEATURE)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        imMatches = cv2.drawMatches(Ref, keypoint1, Rot, keypoint2, matches, None)
        
        return imMatches
    
    def calcShift(self, channel, MAX_FEATURE = 1000, GOOD_MATCH_PERCENT = 0.20, DeltaAngle = 5, step = 0.1, pixelthreshold = 5, margin=256, scorethreshold =0.8, part=False, mode='SIFT'):
        if self.flattenOrder == None:
            self.flatten(channel, order=1)
            print("Flattening is done, order = {}".format(1))
        
        self.angle = self.calcAngle(self.flattenRef, self.flattenRot, self.initangle, DeltaAngle, step, pixelthreshold, margin, scorethreshold)
        keypoint1, keypoint2, matches, _, _ = self.CalcMatches(channel, MAX_FEATURE, part, mode)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        self.trans, theta = self.calcTrans(keypoint1, keypoint2, matches)[:2]
        self.angle = -self.angle+theta
        return self.trans, self.angle

    @staticmethod
    def calcAngle(imgRef, imgRot, DefinedAngle, DeltaAngle=5, step=0.1, threshold=5, margin=256, scorethreshold=0.8):
        count = 0
        imgRefEdge = AutoCanny(imgRef)
        for angle in np.arange(DefinedAngle - DeltaAngle, DefinedAngle + DeltaAngle, step):
            TempImg = imutils.rotate(imgRot, angle)
            TempImgEdge = AutoCanny(TempImg)
            score = CompareEdge(TempImgEdge, imgRefEdge, threshold, margin)

            if score > scorethreshold:
                scorethreshold = score
                angleResult = angle
                count += 1
                print('Angle value is saved.')
        if count == 0:
            angleResult = DefinedAngle

        return angleResult

    @staticmethod
    def calcTrans(keypt1, keypt2, matches):
        points1, points2 = np.zeros((len(matches), 2), dtype=np.float32), np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypt1[match.queryIdx].pt
            points2[i, :] = keypt2[match.trainIdx].pt
        homography = cv2.findHomography(points1, points2, cv2.RANSAC)[0][:2]
        homography_decom = DecomposeHomography(homography)
        return homography_decom

def AutoCanny(image, sigma=0.33):
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def DecomposeHomography(mat):
    a = mat[0,0]
    b = mat[0,1]
    c = mat[0,2]
    d = mat[1,0]
    e = mat[1,1]
    f = mat[1,2]

    p = math.sqrt(a**2 + b**2)
    r = (a*e - b*d)/p
    q = (a*d + b*e)/(a*e - b*d)

    trans = (c, f) #Vertical, Horizontal
    scale = (p, r)
    shear = q
    theta = -math.atan2(b,a) * 180 / math.pi

    return trans, theta, scale, shear

# Compute precision and recall given contours
#@numba.njit
def calc_precision_recall(contours_a, contours_b, threshold):

    count = 0
    for b in range(len(contours_b)):
        # find the nearest distance
        for a in range(len(contours_a)):
            distance = (contours_a[a][0]-contours_b[b][0])**2 + (contours_a[a][1]-contours_b[b][1]) **2

            if distance < threshold **2:
                count = count + 1
                break

    if count != 0:
        precision_recall = count/len(contours_b)
    else:
        precision_recall = 0

    return precision_recall, count, len(contours_b)

def CompareEdge(edge, ref_edge, threshold=5, margin=None):
    # Center location
    x_mid, y_mid = ref_edge.shape[0]/2, ref_edge.shape[1]/2
    # Half size
    if margin == None:
        margin = ref_edge.shape[0]/2
    # Calculate contours
    tmp = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    edge_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    edge_contour = [edge_contours[i][j][0].tolist() for i in range(len(edge_contours)) for j in range(len(edge_contours[i]))]
    edge_contour = [coord for coord in edge_contour if 
                    abs(coord[0]-x_mid) < margin and abs(coord[1] - y_mid) < margin]
    tmp = cv2.findContours(ref_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ref_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    ref_contour = [ref_contours[i][j][0].tolist() for i in range(len(ref_contours)) for j in range(len(ref_contours[i]))]
    ref_contour = [coord for coord in ref_contour if 
                    abs(coord[0]-x_mid) < margin and abs(coord[1] - y_mid) < margin]

    ref_contour = np.int_(np.array(ref_contour))
    edge_contour = np.int_(np.array(edge_contour))

    precision = calc_precision_recall(
            ref_contour, edge_contour, np.array(threshold))[0]    # Precision
        #print("\tprecision:", denominator, numerator)

    recall= calc_precision_recall(
        edge_contour, ref_contour, np.array(threshold))[0]    # Recall
    #print("\trecall:", denominator, numerator)
    
    #print("\trecall:", denominator, numerator)
    if precision == 0 or recall == 0:
        f1 = np.nan
    else:
        f1 = 2*recall*precision/(recall+precision)

    return f1
