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


def lucas_kanade(i, j, window_size, image_x, image_y, image_t, gaussian_kernel):
    '''
    lucas kanade
    '''

    half_window = window_size//2
    i_0 = i-half_window
    i_1 = i+half_window+1
    j_0 = j-half_window
    j_1 = j+half_window+1

    local_x = np.array(image_x[i_0: i_1, j_0: j_1], dtype=float)
    local_y = np.array(image_y[i_0: i_1, j_0: j_1], dtype=float)
    local_t = np.array(image_t[i_0: i_1, j_0: j_1], dtype=float)

    local_x = (local_x*gaussian_kernel).T
    local_y = (local_y*gaussian_kernel).T
    local_t = (local_t*gaussian_kernel).T

    local_x = local_x.flatten(order='F')
    local_y = local_y.flatten(order='F')
    local_t = local_t.flatten(order='F')

    left_matrix = np.array([
        [np.sum(local_x**2), np.sum(local_x*local_y)],
        [np.sum(local_x*local_y), np.sum(local_y**2)]
    ], dtype=float)
    left_matrix = np.linalg.pinv(left_matrix)
    right_matrix = np.array([
        [np.sum(local_x*local_t)],
        [np.sum(local_y*local_t)]
    ], dtype=float)

    velocity_vector = np.matmul(left_matrix, -1*right_matrix).flatten()
    print(f'Ix*u + Iy*v + It = 0 => {np.sum(local_x)}*{velocity_vector[0]} + {np.sum(local_y)}*{velocity_vector[1]} + {np.sum(local_t)} = {np.sum(local_x)*velocity_vector[0] + np.sum(local_y)*velocity_vector[1] + np.sum(local_t)}')
    if np.sum(local_x)*velocity_vector[0] + \
        np.sum(local_y)*velocity_vector[1] + np.sum(local_t) < 1e-1:
        return velocity_vector[1], velocity_vector[0]
    return 0, 0


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

    # derivative with respect to time
    image_t = cv2.GaussianBlur(after_gray, (3*sigma, 3*sigma), sigma) - \
        cv2.GaussianBlur(before_gray, (3*sigma, 3*sigma), sigma)

    # normalize pixels to fit between 0 and 1
    image_x = image_x/255
    image_y = image_y/255
    image_t = image_t/255

    # get points of interest
    # in order to make it harder for everyone
    # opencv decided to switch rows and columns :)
    feature_params = dict(maxCorners=0, qualityLevel=.3,
                          minDistance=7, blockSize=7)
    points_of_interest = cv2.goodFeaturesToTrack(
        before_gray, mask=None, **feature_params)[:, 0, :]
    points_of_interest = np.array(points_of_interest, dtype=int)



    colors = np.random.randint(0, 255, (len(points_of_interest), 3))
    gaussian_kernel = gaus(window_size, sigma)
    
    # TODO: MIXED UP ROWS AND COLUMNS
    for point, color in zip(points_of_interest, colors):
        if point[0] <= window_size//2 or point[0] >= len(before_gray[0])-window_size//2 or point[1] <= window_size//2 or point[1] >= len(before_gray)-window_size//2:
            print(f'rejected ({point[0]}, {point[1]})', end=' because ')
            if point[0] <= window_size//2:
                print(
                    f'x low {point[0]} <= {window_size//2} --> {window_size//2-point[0]}')
            elif point[0] >= len(before_gray[0])-window_size//2:
                print(
                    f'x high {point[0]} >= {len(before_gray[0])-window_size//2} --> {point[0]-len(before_gray[0])+window_size//2}')
            else:
                print('no issue in rows')
            if point[1] <= window_size//2:
                print(
                    f'y low {point[1]} <= {window_size//2} --> {window_size//2-point[1]}')
            elif point[1] >= len(before_gray)-window_size//2:
                print(
                    f'y high {point[1]} >= {len(before_gray)-window_size//2} --> {point[1]-len(before_gray)+window_size//2}')
            else:
                print('no issue in cols')
            continue
        else:
            print(f'rows: {point[0]} < {len(image_x[0])} . . . cols: {point[1]} < {len(image_x)}')

        u,v = lucas_kanade(point[1], point[0], window_size, image_x, image_y, image_t, gaussian_kernel)
        print(f'u,v = ({u}, {v})')
        after_color = cv2.circle(after_color,
                        (int(point[0] + u), int(point[1] + v)),
                        4,
                        color.tolist(),
                        cv2.FILLED)
        after_color = cv2.line(after_color, (point[0], point[1]), (int(point[0] + u), int(point[1] + v)), (color+50).tolist(), 2)
    print(np.shape(image_x))
    return after_color

# from luc_kan1 import optical_flow as opt_flow

def main():
    '''
    main
    '''

    out1 = optical_flow('basketball1.png', 'basketball2.png', window_size=15)

    cv2.imshow('window 1', out1)
    # cv2.imshow('window 2', out2)
    # cv2.imshow('lucas kanade', opt_flow('basketball1.png', 'basketball2.png'))

    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
