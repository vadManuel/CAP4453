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


def my_lucas_kanade(before, after, window_size=5, sigma=3):
    # read images
    before_color = cv2.imread(before)
    after_color = cv2.imread(after)
    before_gray = cv2.imread(before, 0)
    after_gray = cv2.imread(after, 0)

    # convert from multi-color channels to sigle channel gray

    # normalize pixels 0-1

    # derivatives of each image with respect to x and y axes
    before_x = cv2.Sobel(before_gray, -1, 1, 0, 3)
    before_y = cv2.Sobel(before_gray, -1, 0, 1, 3)
    after_x = cv2.Sobel(after_gray, -1, 1, 0, 3)
    after_y = cv2.Sobel(after_gray, -1, 0, 1, 3)

    # mean of derivatives
    image_x = (before_x+after_x)/2
    image_y = (before_y+after_y)/2

    before_gray_gaus = cv2.GaussianBlur(before_gray, (3*sigma, 3*sigma), sigma)
    after_gray_gaus = cv2.GaussianBlur(after_gray, (3*sigma, 3*sigma), sigma)

    # derivate with respect to time
    image_t = after_gray_gaus - before_gray_gaus

    image_x = image_x/255
    image_y = image_y/255
    image_t = image_t/255

    u = np.zeros_like(before_gray)
    v = np.zeros_like(before_gray)

    half_window = window_size//2
    for i in range(half_window, len(before_gray)-half_window):
        for j in range(half_window, len(before_gray[0])-half_window):
            im_x = image_x[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            im_y = image_y[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            im_t = image_t[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            b_matrix = np.reshape(im_t, (len(im_t), 1))
            a_matrix = np.vstack((im_x, im_y)).T

            if np.min(abs(np.linalg.eigvals(np.matmul(a_matrix.T, a_matrix)))) >= 1e-1:
                nu = np.matmul(np.linalg.pinv(a_matrix), b_matrix)
                u[i, j] = nu[0]
                v[i, j] = nu[1]

            '''
            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]
            '''

    return u, v

    # points of interest
    # feature_params = dict(maxCorners=0, qualityLevel=0.01,
    #                       minDistance=1, blockSize=5)
    # poi = cv2.goodFeaturesToTrack(before_gray, mask=None, **feature_params)

    # for point in poi:
    #     left_matrix = np.zeros((2,2))
    #     right_matrix = np.zeros((2,1))

    #     window_row = [point[0] - window_size, point[0] + window_size]
    #     window_col = [point[1] - window_size, point[1] + window_size]
    #     if window_row[0] < 0:
    #         window_row[0] = 0
    #     if window_col[0] < 0:
    #         window_col[0] = 0
    #     if window_row[1] > len(before_gray):
    #         window_row[1] = len(before_gray)
    #     if window_col[1] > len(before_gray[0]):
    #         window_col[1] = len(before_gray[0])

    #     aoi_x = image_x[window_row[0]:window_row[1], window_col[0]:window_col[1]]
    #     aoi_y = image_y[window_row[0]:window_row[1], window_col[0]:window_col[1]]
    #     aoi_t = image_t[window_row[0]:window_row[1], window_col[0]:window_col[1]]
        
    # return image_x, image_y, image_t


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

    out1, out2 = my_lucas_kanade('basketball1.png', 'basketball2.png')

    cv2.imshow('window 1', out1)
    cv2.imshow('window 2', out2)
    # cv2.imshow('window 3', out3)

    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
