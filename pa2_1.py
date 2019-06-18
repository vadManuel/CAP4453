'''
Author: Manuel Vasquez
Date:   03/19/2019


Index:
1. Optical Flow
    a. Lucas-Kanade Optical Flow Estimation
    b.




PIL, Matplotlib, Numpy, Scipy, LibSVM, OpenCV, VLFeat, python-graph
'''


import numpy as np
import cv2
from torchvision import datasets, transforms
import torch
from matplotlib import pyplot as plt


def my_lucas_kanade(before, after):
    # read images
    before = cv2.imread(before)
    after = cv2.imread(after)

    # convert from multi-color channels to sigle channel gray
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=1, blockSize=5)

    # normalize pixels 0-1
    # before_gray = before_gray/255
    # after_gray = after_gray/255

    # derivatives of each image with respect to x and y axes
    image_x = cv2.Sobel(before_gray, -1, 1, 0, 3) - cv2.Sobel(after_gray, -1, 1, 0, 3)
    image_y = cv2.Sobel(before_gray, -1, 0, 1, 3) - cv2.Sobel(after_gray, -1, 0, 1, 3)
    image_t = np.subtract(before_gray, after_gray)

    denominator = np.sum(image_x**2)*np.sum(image_y**2) - \
        np.sum(image_x*image_y)**2

    velocity_x = (np.sum(image_y*image_t)*np.sum(image_x*image_y) - np.sum(image_x*image_t) *
                  np.sum(image_y**2)) / denominator

    velocity_y = (np.sum(image_x*image_t)*np.sum(image_x*image_y) - np.sum(image_y*image_t) *
                  np.sum(image_x**2)) / denominator

    features = cv2.goodFeaturesToTrack(before_gray, mask=None, **feature_params)
    features = np.array(features[:, 0, :], dtype=int)

    

    colors = np.random.randint(0, 255, (len(features), 3))

    for feature, color in zip(features, colors):
        before = cv2.circle(before, (feature[0], feature[1]), 2, color.tolist(), -1)
    
    points = cv2.Sobel(after, -1, 1, 0, 3)*velocity_x + \
        cv2.Sobel(after, -1, 0, 1, 3)*velocity_y

    return points


def lucas_kanade(before, after):
    '''
    lucas kanade
    '''

    feature_params = dict(maxCorners=0, qualityLevel=.3,
                          minDistance=5, blockSize=5)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    initial_frame = cv2.imread(before)
    initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    points_0 = cv2.goodFeaturesToTrack(
        initial_gray, mask=None, **feature_params)
    mask = np.zeros_like(initial_frame)

    next_frame = cv2.imread(after)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    points_1, status, _ = cv2.calcOpticalFlowPyrLK(
        initial_gray, next_gray, points_0, None, **lk_params)

    good_next = points_1[status == 1]
    good_initial = points_0[status == 1]

    for i, (i_next, i_initial) in enumerate(zip(good_next, good_initial)):
        line_start = tuple(i_next.ravel())
        line_end = tuple(i_initial.ravel())
        mask = cv2.line(mask, line_start, line_end, color[i].tolist(), 2)
        next_frame = cv2.circle(next_frame, line_start,
                                4, color[i].tolist(), -1)
    img = cv2.add(next_frame, mask)

    return img


def main():
    '''
    main
    '''

    out1 = my_lucas_kanade(
        'basketball1.png', 'basketball2.png')

    cv2.imshow('window 1', out1)
    # cv2.imshow('window 2', before_y)

    '''
    before = cv2.imread('basketball1.png')
    after = cv2.imread('basketball2.png')
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    # before = before / 255

    # before = before[:,:,0]

    feature_params = dict(maxCorners=0,
                          qualityLevel=0.01,
                          minDistance=1,
                          blockSize=5)

    out1 = None
    features = cv2.goodFeaturesToTrack(before_gray, mask=None, **feature_params)
    features = np.array(features[:, 0, :], dtype=int)
    colors = np.random.randint(0, 255, (len(features), 3))
    
    for feature, color in zip(features, colors):
        before = cv2.circle(before, (feature[0], feature[1]), 2, color.tolist(), -1)
    '''


    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
