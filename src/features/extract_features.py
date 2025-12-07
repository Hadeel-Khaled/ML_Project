import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog


def feature_vector(img):
    #return a 1D vector represents the pic
    #convert the pic to gray using cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #HOG for picture edges اتجاهات and divide the picture into cells, extract histogram for each cell, then collect all histograms into one vector
    hog_feat = hog(
        gray,
        orientations=12, #directions
        pixels_per_cell=(8, 8), # divide each cell into 8*8 pixel (جواها اتجاهات هنجمعها و نعملها هستوجرام )
        cells_per_block=(2, 2),
        feature_vector=True
    )

    #color information (HSV) H = Hue → S = Saturation → V = Value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, [16, 16, 16],
        [0, 180, 0, 256, 0, 256]
    ).flatten() # converting from 3D -> 8×8×8 = 512 ، to 1D -> 512
    hist = cv2.normalize(hist, hist).flatten()

    lbp = local_binary_pattern(gray, P=24, R=3)
    (lbphist, _) = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbphist = lbphist.astype("float")
    lbphist /= lbphist.sum()

    features = np.concatenate([hog_feat, hist, lbphist]) #عشان نجمع الشكل واللون
    return features


