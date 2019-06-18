'''
Author: Manuel Vasquez
Date:   03/19/2019


Index:
1. Optical Flow
    a. Lucas-Kanade Optical Flow Estimation
    b.




PIL, Matplotlib, Numpy, Scipy, LibSVM, OpenCV, VLFeat, python-graph
'''


import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def gaus(window_size, sigma):
    '''
    gaus
    '''

    half_window = window_size//2
    output_kernel = np.zeros((window_size, window_size))
    output_kernel[half_window, half_window] = 1
    return gaussian_filter(output_kernel, sigma)


def lucas_kanade(i, j, window_size, image_x, image_y, image_t):
    '''
    lucas kanade
    '''

    half_window = window_size//2
    i_0 = i-half_window if i-half_window >= 0 else 0
    i_1 = i+half_window+1 if i+half_window+1 <= len(image_x) else len(image_x)
    j_0 = j-half_window if j-half_window >= 0 else 0
    j_1 = j+half_window+1 if j+half_window+1 <= len(image_x[0]) else len(image_x[0])

    local_x = image_x[i_0 : i_1, j_0 : j_1].T
    local_y = image_y[i_0 : i_1, j_0 : j_1].T
    local_t = image_t[i_0 : i_1, j_0 : j_1].T

    local_x = local_x.flatten(order='F')
    local_y = local_y.flatten(order='F')
    local_t = -1*local_t.flatten(order='F')

    # b_matrix = np.reshape(im_t, (len(im_t), 1))
    # a_matrix = np.vstack((im_x, im_y)).T

    A = np.vstack((local_x, local_y)).T
    # U = np.dot(np.dot(np.linalg.pinv(A), A.T), local_t)

    if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= 1e-1:
        U = np.matmul(np.linalg.pinv(A), local_t)
        return U[0], U[1]
    return 0, 0
    
    # np.matmul(np.linalg.pinv(a_matrix), b_matrix)
    # U = np.dot(np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T), local_t)

    # if np.min(abs(np.linalg.eigvals(np.matmul(a_matrix.T, a_matrix)))) >= 1e-1:

    # return U[0], U[1]

    # return src


def optical_flow(before, after, window_size=5, sigma=1):
    '''
    optical flow
    '''

    # read images
    before_color = cv2.imread(before)
    after_color = cv2.imread(after)
    before_gray = cv2.imread(before, 0)
    after_gray = cv2.imread(after, 0)

    # derivatives of each image with respect to x and y axes
    before_x = cv2.Sobel(before_gray, -1, 1, 0, 3)
    before_y = cv2.Sobel(before_gray, -1, 0, 1, 3)
    after_x = cv2.Sobel(after_gray, -1, 1, 0, 3)
    after_y = cv2.Sobel(after_gray, -1, 0, 1, 3)

    # mean of derivatives
    image_x = (before_x + after_x)/2
    image_y = (before_y + after_y)/2

    # preprocess (blur) to calculate derivate with respect to time
    # before_gray_gaus = cv2.GaussianBlur(before_gray, (3*sigma, 3*sigma), sigma)
    # after_gray_gaus = cv2.GaussianBlur(after_gray, (3*sigma, 3*sigma), sigma)

    # derivative with respect to time
    # image_t = after_gray_gaus - before_gray_gaus
    image_t = after_gray - before_gray

    # normalize pixels to fit between 0 and 1
    image_x = image_x/255
    image_y = image_y/255
    image_t = image_t/255

    # get points of interest
    feature_params = dict(maxCorners=0, qualityLevel=.3,
                          minDistance=7, blockSize=7)
    points_of_interest = cv2.goodFeaturesToTrack(
        before_gray, mask=None, **feature_params)[:, 0, :]
    points_of_interest = np.array(points_of_interest, dtype=int)

    colors = np.random.randint(0, 255, (len(points_of_interest), 3))

    for point, color in zip(points_of_interest, colors):
        u, v = lucas_kanade(point[0], point[1],
                            window_size, image_x, image_y, image_t)
        after_color = cv2.circle(after_color, (int(point[0] + u),
                                               int(point[1] + v)), 4, color.tolist(), cv2.FILLED)
        after_color = cv2.line(after_color, (point[0], point[1]),
                               (int(point[0] + u), int(point[1] + v)), (color+50).tolist(), 2)

    return before_color-after_color, after_color

    # u = np.zeros_like(before_gray)
    # v = np.zeros_like(before_gray)

    # half_window = window_size//2
    # for i in range(half_window, len(before_gray)-half_window):
    #     for j in range(half_window, len(before_gray[0])-half_window):
    #         im_x = image_x[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
    #         im_y = image_y[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
    #         im_t = image_t[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
    #         b_matrix = np.reshape(im_t, (len(im_t), 1))
    #         a_matrix = np.vstack((im_x, im_y)).T

    #         if np.min(abs(np.linalg.eigvals(np.matmul(a_matrix.T, a_matrix)))) >= 1e-1:
    #             nu = np.matmul(np.linalg.pinv(a_matrix), b_matrix)
    #             u[i, j] = nu[0]
    #             v[i, j] = nu[1]

    # return u, v

from luc_kan1 import optical_flow as opt_flow

def main():
    '''
    main
    '''

    # GAUSSIAN_KERNEL = gaus(5, 3)
    out1, out2 = optical_flow('basketball1.png', 'basketball2.png', window_size=15)

    cv2.imshow('window 1', out1)
    cv2.imshow('window 2', out2)
    cv2.imshow('lucas kanade', opt_flow('basketball1.png', 'basketball2.png'))

    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
