import numpy as np
import cv2
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from img_proc import *
from img_io import *


def _contrast_adjustment():
    parser = argparse.ArgumentParser()

    # ''' img1 '''
    # parser.add_argument('--img_file', type=str, default='img/man.jpg')
    # parser.add_argument('--is_gray', type=bool, default=True)
    # parser.add_argument('--output_file', type=str, default='output/man_{c:02d}_{gamma_id:02d}')
    
    # ''' img2 '''
    # parser.add_argument('--img_file', type=str, default='img/street.jpg')
    # parser.add_argument('--is_gray', type=bool, default=True)
    # parser.add_argument('--output_file', type=str, default='output/street_{c:02d}_{gamma_id:02d}')

    ''' img3 '''
    parser.add_argument('--img_file', type=str, default='img/deers.png')
    parser.add_argument('--is_gray', type=bool, default=True)
    parser.add_argument('--output_file', type=str, default='output/deers_{c:02d}_{gamma_id:02d}')

    # ''' img1 '''
    # parser.add_argument('--img_file', type=str, default='img/red_flower.jpg')
    # parser.add_argument('--is_gray', type=bool, default=False)
    # parser.add_argument('--output_file', type=str, default='output/red_flower_{c:02d}_{gamma_id:02d}')

    # ''' img2 '''
    # parser.add_argument('--img_file', type=str, default='img/yellow_flower.jpg')
    # parser.add_argument('--is_gray', type=bool, default=False)
    # parser.add_argument('--output_file', type=str, default='output/yellow_flower_{c:02d}_{gamma_id:02d}')

    # ''' img3 '''
    # parser.add_argument('--img_file', type=str, default='img/rose.jpg')
    # parser.add_argument('--is_gray', type=bool, default=True)
    # parser.add_argument('--output_file', type=str, default='output/rose_{c:02d}_{gamma_id:02d}')
    


    args = parser.parse_args()

    img = read_img(args.img_file, args.is_gray)

    ''' median filter '''
    # img = median_filter(img, 3)
    # display_img(img, args.is_gray)

    ''' bilateral filter '''
    print(img.shape)
    img_new = bilateral_filter(img, 5, 4, 4)
    display_img(img_new, args.is_gray)
    img_2 = cv2.bilateralFilter(img, 9, 100, 100)
    display_img(img_2, args.is_gray)

    ''' power law transformation '''
    # const = [0.8, 1, 3/2]
    # gamma = [0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5.0, 10.0]
    # for c in const:
    #     for i in range(len(gamma)):
    #         print(args.output_file.format(c=int(c*10), gamma_id=i))
    #         print(gamma[i])
    #         img_corrected = power_law_trans(img, gamma[i], c)
    #         save_img(img_corrected, args.output_file.format(c=int(c*10), gamma_id=i))

    ''' smoothing filter '''
    # kernel = get_kernel1d(kernel_size=3, type='gaussian')
    # img = conv_separable(img, kernel)
    # display_img(img, args.is_gray)
    

    ''' histogram equilization '''
    # if args.is_gray:
    #     img = equalize_hist(img)
    #     display_img(img, args.is_gray)
    # else:
    #     img_corrected = img
    #     # img_corrected = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     img_corrected[:, :, 0] = equalize_hist(img_corrected[:, :, 0])
    #     img_corrected[:, :, 1] = equalize_hist(img_corrected[:, :, 1])
    #     img_corrected[:, :, 2] = equalize_hist(img_corrected[:, :, 2])
    #     # img_corrected = cv2.cvtColor(img_corrected, cv2.COLOR_HSV2BGR)
    #     display_img(img_corrected, args.is_gray)
        
    ''' laplacian filter '''
    # kernel = get_kernel2d(type='laplacian_sharpen')
    # img = conv_nonseparable(img, kernel)
    # display_img(img, args.is_gray)


def experiment_1():
    ''' Perform contrast adjustment on overexposure image.
        Here, an image of a man is used. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_file', type=str, default='img/man.jpg')
    parser.add_argument('--is_gray', type=bool, default=True)
    parser.add_argument('--output_file', type=str, default='output_man_gaussian/man')
    # parser.add_argument('--output_file', type=str, default='output_man_no_blur/man')

    args = parser.parse_args()
    img = read_img(args.img_file, args.is_gray)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=3, type='gaussian')
    img_new = conv_separable(img, kernel)
    display_img(img_new, args.is_gray)

    ''' laplacian filter '''
    kernel = get_kernel2d(type='laplacian_2')
    img_new = conv_nonseparable(img_new, kernel).astype('int64')
    img = img.astype('int64') + 2 * img_new
    img[img > 255] = 255; img[img < 0] = 0
    display_img(2 * img_new, args.is_gray)

    ''' bilateral filter '''
    img = bilateral_filter(img, 5, 2, 2)
    display_img(img, args.is_gray)

    ''' power law transformation '''
    output_file = args.output_file+'_{c:02d}_{gamma_id:02d}'
    const = [0.8, 1, 1.2]
    gamma = [1, 1.5, 2.5, 5.0]
    for c in const:
        for i in range(len(gamma)):
            print(output_file.format(c=int(c*10), gamma_id=i))
            print(gamma[i])
            img_corrected = power_law_trans(img, gamma[i], c)
            save_img(img_corrected, output_file.format(c=int(c*10), gamma_id=i))

    ''' median filter '''
    img = median_filter(img, 3)
    display_img(img, args.is_gray)

    ''' histogram equilization '''
    if args.is_gray:
        img = equalize_hist(img)
        print(args.output_file)
        display_img(img, args.is_gray)
        save_img(img, args.output_file)


def experiment_2():
    ''' Perform contrast adjustment on underexposure image.
        Here, an image of a plant is used. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_file', type=str, default='img/plant.png')
    parser.add_argument('--is_gray', type=bool, default=True)
    parser.add_argument('--output_file', type=str, default='output_plant/plant')

    args = parser.parse_args()
    img = read_img(args.img_file, args.is_gray)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=3, type='gaussian')
    img_new = conv_separable(img, kernel)
    display_img(img_new, args.is_gray)

    ''' laplacian filter '''
    kernel = get_kernel2d(type='laplacian_2')
    img_new = conv_nonseparable(img_new, kernel).astype('int64')
    img = img.astype('int64') + 2 * img_new
    img[img > 255] = 255
    img[img < 0] = 0
    display_img(2 * img_new, args.is_gray)

    ''' bilateral filter '''
    img = bilateral_filter(img, 5, 2, 2)
    display_img(img, args.is_gray)

    ''' power law transformation '''
    output_file = args.output_file+'_{c:02d}_{gamma_id:02d}'
    const = [0.8, 1, 1.2]
    gamma = [0.4, 0.7, 1]
    for c in const:
        for i in range(len(gamma)):
            print(output_file.format(c=int(c*10), gamma_id=i))
            print(gamma[i])
            img_corrected = power_law_trans(img, gamma[i], c)
            save_img(img_corrected, output_file.format(c=int(c*10), gamma_id=i))

    ''' median filter '''
    # img = median_filter(img, 3)
    # display_img(img, args.is_gray)

    ''' histogram equilization '''
    if args.is_gray:
        output_file = args.output_file+'_hist'
        img = equalize_hist(img)
        print(args.output_file)
        display_img(img, args.is_gray)
        save_img(img, output_file)


def experiment_3():
    ''' Perform noise removal. 
        Here, an image of deers is used. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_file', type=str, default='img/deers.png')
    parser.add_argument('--is_gray', type=bool, default=True)
    parser.add_argument('--output_file', type=str, default='output_deers/deers')

    args = parser.parse_args()
    img = read_img(args.img_file, args.is_gray)

    ''' median filter '''
    img_med = median_filter(img, 3)
    display_img(img_med, args.is_gray)
    output_file = args.output_file+'_median'
    save_img(img_med, output_file)

    ''' bilateral filter '''
    img_bi = bilateral_filter(img, 5, 3, 3)
    display_img(img_bi, args.is_gray)
    output_file = args.output_file+'_bi_k5_s3_v3'
    save_img(img_bi, output_file)

    ''' bilateral filter '''
    img_bi = bilateral_filter(img, 5, 3, 5)
    display_img(img_bi, args.is_gray)
    output_file = args.output_file+'_bi_k5_s3_v5'
    save_img(img_bi, output_file)

    ''' bilateral filter '''
    img_bi = bilateral_filter(img, 5, 10, 3)
    display_img(img_bi, args.is_gray)
    output_file = args.output_file+'_bi_k5_s10_v3'
    save_img(img_bi, output_file)
    
    ''' bilateral filter '''
    img_bi = bilateral_filter(img, 7, 10, 3)
    display_img(img_bi, args.is_gray)
    output_file = args.output_file+'_bi_k7_s10_v3'
    save_img(img_bi, output_file)


    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=3, type='gaussian')
    img_gaussian = conv_separable(img, kernel)
    display_img(img_gaussian, args.is_gray)
    output_file = args.output_file+'_gaussian_3'
    save_img(img_gaussian, output_file)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=5, type='gaussian')
    img_gaussian = conv_separable(img, kernel)
    display_img(img_gaussian, args.is_gray)
    output_file = args.output_file+'_gaussian_5'
    save_img(img_gaussian, output_file)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=3, type='averaging')
    img_box = conv_separable(img, kernel)
    display_img(img_box, args.is_gray)
    output_file = args.output_file+'_box_3'
    save_img(img_box, output_file)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=5, type='averaging')
    img_box = conv_separable(img, kernel)
    display_img(img_box, args.is_gray)
    output_file = args.output_file+'_box_5'
    save_img(img_box, output_file)


def experiment_4():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_file', type=str, default='img/rose.jpg')
    parser.add_argument('--is_gray', type=bool, default=False)
    parser.add_argument('--output_file', type=str, default='output_rose/rose')
    
    args = parser.parse_args()

    img = read_img(args.img_file, args.is_gray)
    img_med = np.empty(img.shape)
    img_bi = np.empty(img.shape)
    img_gaussian = np.empty(img.shape)
    img_box = np.empty(img.shape)
    

    ''' median filter '''
    img_med[:, :, 0] = median_filter(img[:, :, 0], 5)
    img_med[:, :, 1] = median_filter(img[:, :, 1], 5)
    img_med[:, :, 2] = median_filter(img[:, :, 2], 5)
    
    img_med = img_med.astype('uint8')
    display_img(img_med, args.is_gray)
    output_file = args.output_file+'_median'
    save_img(img_med, output_file)

    ''' bilateral filter '''
    img_bi[:, :, 0] = bilateral_filter(img[:, :, 0], 5, 3, 3)
    img_bi[:, :, 1] = bilateral_filter(img[:, :, 1], 5, 3, 3)
    img_bi[:, :, 2] = bilateral_filter(img[:, :, 2], 5, 3, 3)
    
    img_bi = img_bi.astype('uint8')
    display_img(img_bi, args.is_gray)
    output_file = args.output_file+'_bi'
    save_img(img_bi, output_file)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=5, type='gaussian')
    img_gaussian[:, :, 0] = conv_separable(img[:, :, 0], kernel)
    img_gaussian[:, :, 1] = conv_separable(img[:, :, 1], kernel)
    img_gaussian[:, :, 2] = conv_separable(img[:, :, 2], kernel)
    img_gaussian = img_gaussian.astype('uint8')
    display_img(img_gaussian, args.is_gray)
    output_file = args.output_file+'_gaussian_5'
    save_img(img_gaussian, output_file)

    ''' smoothing filter '''
    kernel = get_kernel1d(kernel_size=5, type='averaging')
    img_box[:, :, 0] = conv_separable(img[:, :, 0], kernel)
    img_box[:, :, 1] = conv_separable(img[:, :, 1], kernel)
    img_box[:, :, 2] = conv_separable(img[:, :, 2], kernel)
    img_box = img_box.astype('uint8')
    display_img(img_box, args.is_gray)
    output_file = args.output_file+'_box_5'
    save_img(img_box, output_file)


if __name__ == '__main__':
    experiment_1()