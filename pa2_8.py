'''
Author: Manuel Vasquez
Date:   03/19/2019

Python 3.6.5 64-bit (Anaconda3)
'''


import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def gaussian_window(window_size, sigma):
    '''
    Create a gaussian window to weigh pixels surrounding center point of window.
    '''

    half_window = window_size//2
    output_kernel = np.zeros((window_size, window_size))
    output_kernel[half_window, half_window] = 1

    return gaussian_filter(output_kernel, sigma)


def lucas_kanade(center_row, center_col, window_size, weighted_window, img_before_rows, img_before_cols, img_time):
    '''
    Solve for variable u and v by getting windows from each derivative x, y, and t.
    u = (I_yt*I_xy - I_xt*I_yy)/(I_xx*I_yy - (I_xy)^2)
    v = (I_xt*I_xy - I_yt*I_xx)/(I_xx*I_yy - (I_xy)^2)
    '''
    
    # utility
    half_window = window_size//2

    # get window around center point
    window_rows = weighted_window*img_before_rows[center_row-half_window : center_row+half_window+1, center_col-half_window : center_col+half_window+1]
    window_cols = weighted_window*img_before_cols[center_row-half_window : center_row+half_window+1, center_col-half_window : center_col+half_window+1]
    window_time = weighted_window*img_time[center_row-half_window : center_row+half_window+1, center_col-half_window : center_col+half_window+1]

    # get velocities for each component
    denominator = np.sum(window_cols**2)*np.sum(window_rows**2) - np.sum(window_cols*window_rows)**2 
    u = (np.sum(window_rows*window_time)*np.sum(window_cols*window_rows) - np.sum(window_cols*window_time)*np.sum(window_rows**2))/denominator
    v = (np.sum(window_cols*window_time)*np.sum(window_cols*window_rows) - np.sum(window_rows*window_time)*np.sum(window_cols**2))/denominator

    return u, v

def optical_flow(before, after, window_size=7, sigma=1, save=False):
    '''
    Utilize cv2 function goodFeaturesToTrack with the feature parameters
    indicated below track between both images using the Lucas Kanade
    method.
    The parameter sigma, in this case, is used for the generation of the
    gaussian window, the points further away from the center would be
    the most penalized.
    I found that by using the Scharr method instead of Sobel, I could
    track features better.
    '''

    # read images
    img_before = cv2.imread(before)
    img_after = cv2.imread(after)

    # convert to gray scale
    img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_after_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

    # randomly generated colors
    colors = np.random.randint(0, 255, (len(img_after), 3))

    # parameters for goodFeaturesToTrack 
    feature_params = dict(maxCorners=0, qualityLevel=.3, minDistance=7, blockSize=7)

    # features to track
    poi = cv2.goodFeaturesToTrack(img_before_gray, mask=None, **feature_params)

    # normalizing
    img_before_gray = img_before_gray/255
    img_after_gray = img_after_gray/255

    # calculate derivatives with respect to rows and cols
    img_before_rows = cv2.Scharr(img_before_gray, -1, 0, 1)
    img_before_cols = cv2.Scharr(img_before_gray, -1, 1, 0)

    # calculate derivative with respect to time
    img_time = img_after_gray - img_before_gray

    # gaussian kernel to weight pixels relative to center of window
    weighted_window = gaussian_filter(window_size, sigma)

    # draw lines signifying the flow and circles as the endpoint
    for point, color in zip(poi, colors):
        row_0 = int(point[0][1])
        col_0 = int(point[0][0])
        u, v = lucas_kanade(row_0, col_0, window_size,
                            weighted_window, img_before_rows, img_before_cols, img_time)
        row_1 = row_0+int(100*v)
        col_1 = col_0+int(100*u)

        img_after = cv2.circle(img_after, (col_1, row_1), 4, color.tolist(), cv2.FILLED)
        img_after = cv2.line(img_after, (col_0, row_0), (col_1, row_1), (color+50).tolist(), 2)

    # saving image
    if save:
        cv2.imwrite('single_scale_lk.png', img_after)

    return img_after


def main():
    '''
    main
    '''

    out = optical_flow('basketball1.png', 'basketball2.png', save=False)
    cv2.imshow('window', out)

    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
