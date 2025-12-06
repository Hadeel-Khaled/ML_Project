import cv2
import numpy as np
from skimage.feature import hog

def feature_vector(img):
    #return a 1D vector represents the pic
    #convert the pic to gray using cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #HOG for picture edges اتجاهات and divide the picture into cells, extract histogram for each cell, then collect all histograms into one vector
    hog_feat = hog(
        gray,
        orientations=9, #directions
        pixels_per_cell=(16, 16), # divide each cell into 16*16 pixel (جواها اتجاهات هنجمعها و نعملها هستوجرام )
        cells_per_block=(2, 2),
        feature_vector=True
    )

    #color information (HSV) H = Hue → S = Saturation → V = Value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    ).flatten() # converting from 3D -> 8×8×8 = 512 ، to 1D -> 512

    features = np.concatenate([hog_feat, hist]) #عشان نجمع الشكل واللون
    return features

