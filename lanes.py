import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    # Converting into grayscale image (Matrix)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Calculates deravative of x, y axix
    # returns white pixel -> high change in intensity exceeding threshold and black for low change
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Loading image
image = cv2.imread('assets/test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)

cropped_image = region_of_interest(canny)
cv2.imshow('Result', cropped_image)
cv2.waitKey(0)
