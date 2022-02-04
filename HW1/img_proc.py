import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import *


def power_law_trans(img, gamma, c=1.0):
    """ Power-law transformations
    Args:
        img (ndarray): with shape (height, width) for gray / (height, width, channel) for BGR
        gamma (float)
        c (float)
    Returns:
        img_corrected (ndarray): its shape is the same as input's
    """
    img_corrected = np.array(c * 255 * (img / 255) ** gamma)
    img_corrected[img_corrected > 255] = 255
    img_corrected = img_corrected.astype('uint8')
    return img_corrected


def equalize_hist(img):
    """ Histogram equalization
    Args:
        img (ndarray): its shape is (height, width)
    Returns:
        img_corrected (ndarray): its shape is the same as input's
    """
    assert len(img.shape) == 2, 'Input image has more than one channel. '
    count_raw = np.bincount(img.flatten(), minlength=256)
    sum, cnt_average = 0, np.sum(count_raw) / 256
    lookup_table = np.zeros(256)
    for i in range(256):
        sum += count_raw[i]
        lookup_table[i] = int(sum / cnt_average)
    lookup_table[lookup_table > 255] = 255
    
    (height, width) = img.shape
    img_corrected = np.empty(img.shape, dtype='uint8')
    for h in range(height):
        for w in range(width):
            img_corrected[h][w] = lookup_table[img[h][w]]
    # display_hist(count_raw)
    # count_new = np.bincount(img_corrected.flatten(), minlength=256)
    # display_hist(count_new)
    return img_corrected


def get_kernel1d(kernel_size=3, type='averaging'):
    """ Get 1D separable filter
    Args:
        kernel_size (int)
        type (str): type of kernel
    Returns:
        kernel (ndarray): 1D separable kernel
    """
    assert kernel_size % 2 == 1, 'Kernel size is not an odd number. '

    if type == 'averaging':
        return np.ones(kernel_size) / kernel_size
    if type == 'gaussian':
        kernel = np.fromfunction(lambda x: np.exp((-1*(x-(kernel_size-1)/2) ** 2) / 2 ** 2), (kernel_size,))
        return kernel / np.sum(kernel)
    if type == 'laplacian':
        return np.array((-1, 2, -1))

    assert False, 'Input filter type does not exist. '


def conv_separable(img, kernel):
    """ Perform convolution with separable kernel
    Args:
        img (ndarray): its shape is (height, width)
        kernel (ndarray): 1D separable kernel
    Returns:
        img_corrected (ndarray): its shape is the same as input's
    """
    assert len(img.shape) == 2, 'Input image has more than one channel. '

    kernel_size = len(kernel)
    ''' zero padding '''
    (height, width) = img.shape
    img_padding = np.zeros((height+kernel_size-1, width+kernel_size-1))
    origin = int(kernel_size/2)
    img_padding[origin:origin+height, origin:origin+width] = img
    ''' separable convolution '''
    img_temp = np.empty((height+kernel_size-1, width))
    img_corrected = np.empty((height, width))
    for h in range(height+kernel_size-1):
        for w in range(width):
            img_temp[h][w] = np.dot(kernel, img_padding[h, w:w+kernel_size])
    for h in range(height):
        for w in range(width):
            img_corrected[h][w] = np.dot(kernel, img_temp[h:h+kernel_size, w])
    img_corrected[img_corrected > 255] = 255
    img_corrected[img_corrected < 0] = 0

    return img_corrected.astype('uint8')


def get_kernel2d(type='laplacian'):
    """ Get 2D filter
    Args:
        type (str): type of kernel
    Returns:
        kernel (ndarray): 2D kernel
    """
    if type == 'laplacian_1':
        return np.array(((0, -1, 0), (-1, 4, -1), (0, -1, 0)))
    if type == 'laplacian_2':
        return np.array(((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)))
    if type == 'laplacian_sharpen':
        return np.array(((0, -1, 0), (-1, 5, -1), (0, -1, 0)))
    assert False, 'Input filter type does not exist. '


def conv_nonseparable(img, kernel):
    """ Perform convolution with separable kernel
    Args:
        img (ndarray): its shape is (height, width)
        kernel (ndarray): 2D kernel
    Returns:
        img_corrected (ndarray): its shape is the same as input's
    """
    assert len(img.shape) == 2, 'Input image has more than one channel. '
    assert len(kernel.shape) == 2, 'Kernel is not 2D. '

    (kernel_size, _) = kernel.shape
    ''' zero padding '''
    (height, width) = img.shape
    img_padding = np.zeros((height+kernel_size-1, width+kernel_size-1))
    origin = int(kernel_size/2)
    img_padding[origin:origin+height, origin:origin+width] = img
    ''' convolution '''
    img_corrected = np.empty((height, width))
    for h in range(height):
        for w in range(width):
            product = np.multiply(kernel, img_padding[h:h+kernel_size, w:w+kernel_size])
            img_corrected[h][w] = np.sum(product)
    img_corrected[img_corrected > 255] = 255
    img_corrected[img_corrected < 0] = 0

    return img_corrected.astype('uint8')


def median_filter(img, kernel_size=3):
    """ Perform median filtering
    Args:
        img (ndarray): its shape is (height, width)
        kernel_size (int)
    Returns:
        img_corrected (ndarray): its shape is the same as input's
    """
    assert len(img.shape) == 2, 'Input image has more than one channel. '
    ''' zero padding '''
    (height, width) = img.shape
    idx_median = np.floor(kernel_size * kernel_size / 2).astype('int')
    img_padding = np.zeros((height+kernel_size-1, width+kernel_size-1))
    origin = int(kernel_size/2)
    img_padding[origin:origin+height, origin:origin+width] = img
    img_corrected = np.empty((height, width))
    for h in range(height):
        for w in range(width):
            patch = img_padding[h:h+kernel_size, w:w+kernel_size]
            img_corrected[h][w] = np.sort(patch, axis=None)[idx_median]
    
    return img_corrected.astype('uint8')


def bilateral_filter(img, kernel_size, sigma_s, sigma_v):
    """ Perform bilateral filtering
    Args:
        img (ndarray)
        sigma_s (float): spatial gaussian std. dev.
        sigma_v (float): value gaussian std. dev.
    Returns:
        img_corrected (ndarray)
    """
    assert len(img.shape) == 2, 'Input image has more than one channel. '
    img = img
    half_size = np.floor(kernel_size/2).astype('int')
    gaussian = lambda r2, sigma: np.exp( -0.5*r2/sigma**2 )
    weight_sum = np.ones(img.shape) * np.finfo(np.float32).eps
    img_corrected  = np.ones(img.shape) * np.finfo(np.float32).eps

    for shft_x in range(-half_size,half_size+1):
        for shft_y in range(-half_size,half_size+1):
            # compute the spatial weight
            w = gaussian(shft_x**2+shft_y**2, sigma_s)
            # shift by the offsets
            off = np.roll(img, [shft_y, shft_x], axis=[0,1])
            # compute the value weight
            tw = w * gaussian((off-img)**2, sigma_v)
            # accumulate the results
            img_corrected += off * tw
            weight_sum += tw
    # normalize the result and return
    img_corrected = (img_corrected / weight_sum).astype('uint8')
    img_corrected[img_corrected > 255] = 255
    img_corrected[img_corrected < 0] = 0
    return img_corrected
