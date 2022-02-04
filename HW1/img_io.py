import numpy as np
import cv2
import argparse
from pathlib import Path
from matplotlib import pyplot as plt


def read_img(filename, is_gray):
    """
    Args:
        filename (str)
        is_gray (bool)
    Returns:
        img (ndarray): with shape (height, width) for gray / (height, width, channel) for BGR
    """
    img = cv2.imread(filename)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def save_img(img, output_file):
    """ Output PNG file for the image
    Args:
        img (ndarray)
        output_file (str): without file extension
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_file+'.png', img)


def display_img(img, is_gray):
    """ Display the image via plt
    Args:
        img (ndarray): with shape (height, width) for gray / (height, width, channel) for BGR
        is_gray (bool)
    """
    print('The shape of the image: ', img.shape)
    if is_gray:
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()
    plt.close()


def display_hist(count):
    """ Given count of each value, display the histogram via bar chart
    Args:
        count (ndarray of ints): length is 256
    """
    plt.bar(range(256), count, width=1.0)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()
    plt.close()

