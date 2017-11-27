__author__ = "Liran Drory"

import numpy as np
from scipy.signal import convolve2d as conv2d
import matplotlib.pylab as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve as conv
import os


GRAYSCAL = 1
RGB = 2
RGB_MATRIX = 3
WIDTH = 1
SHADES_OF_GRAY = 255

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def read_image(filename, representation):
    """
    Read Image and return matrix [0,1] float64
    Gray scale - 2D
    RGB - 3D

    Parameters
    ----------
    :param filename: str
        string containing the image filename to read (PATH)

    :param representation: int
        either 1 or 2 defining whether the output
        should be a greyscale image (1) or an RGB image (2).

    Returns
    -------
    :return numpy array with either 2D matrix or 3D matrix
            describing the pixels of the image

    """

    # loads the image
    im = imread(filename)

    if representation == RGB:
        im_float = im.astype(np.float64)    # pixels to float
        im_float /= SHADES_OF_GRAY          # pixels [0,1]
        return im_float

    if representation == GRAYSCAL:
        im_g = im.astype(np.float64)       # pixels to float
        im_g = rgb2gray(im_g)                 # turn to grey
        return im_g


def gaussian_kernel(kernel_size):
    """
    create gaussian matrix

    Parameters
    ----------
    :param kernel_size

    Returns
    -------
    :return gaussian matrix

    """

    gaussian = binomial_ker = np.array([[1, 1]])
    while gaussian.shape[1] < kernel_size: gaussian = conv2d(gaussian, binomial_ker)

    return 1 / gaussian.sum() * gaussian


def reduce_im(im, filter_vec):

    # step 1: blur
    im = conv(im, filter_vec)               # convolution with horizontal filter
    im = conv(im, filter_vec.transpose())   # convolution with vertical filter

    # step 2: reduce
    return im[::2, 1::2]


def expand_im(im, filter_vec, expand_shape):

    # step 1: expand
    M, N = expand_shape
    expanded_im = np.zeros((M, N))
    expanded_im[::2, 1::2] = im

    # step 2: blur
    expanded_im = conv(expanded_im, filter_vec)               # convolution with horizontal filter
    expanded_im = conv(expanded_im, filter_vec.transpose())   # convolution with vertical filter

    return expanded_im


def build_gaussian_pyramid(im, max_levels, filter_size):

    # create filter vec
    filter_vec = gaussian_kernel(filter_size)

    pyr = [im]
    for i in range(max_levels - 1):
        pyr.append(reduce_im(pyr[i], filter_vec))

    return [pyr, filter_vec]


def build_laplacian_pyramid(im, max_levels, filter_size):

    # build the gaussian pyramid
    gaus_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    # build the laplacian pyramid
    pyr = []
    for i in range(max_levels-1):
        pyr.append(gaus_pyr[i] - expand_im(gaus_pyr[i+1], filter_vec*2, gaus_pyr[i].shape))
    pyr.append(gaus_pyr[max_levels-1])

    return [pyr, filter_vec]


def laplacian_to_image(lpyr, filter_vec, coeff):

    # multiply each level of laplacian pyr with coeff
    lpyr = [coeff[i]*lpyr[i] for i in range(len(lpyr))]

    if len(lpyr) == 1:
        return lpyr[0]

    if len(lpyr) == 2:
        return lpyr[0] + expand_im(lpyr[1], 2*filter_vec, lpyr[0].shape)

    G = laplacian_to_image(lpyr[1:], filter_vec, coeff)
    return lpyr[0] + expand_im(G, 2*filter_vec, lpyr[0].shape)


def render_pyramid(pyr,levels):
    heigth = pyr[0].shape[0]
    width = sum([pyr[i].shape[1] for i in range(levels)])
    return np.zeros((heigth, width))


def display_pyramid(pyr, levels):

    # get the black_box size to fit all the pyramid
    black_box = render_pyramid(pyr, levels)

    offset = width = 0
    for i in range(levels):
        offset += width
        height, width = pyr[i].shape
        pyr[i] = (pyr[i] - pyr[i].min()) / \
                 (pyr[i].max() - pyr[i].min()) #strech to [0,1]

        black_box[:height, offset:width+offset] = pyr[i]

    plt.imshow(black_box, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, mask_levels, filter_size_im, filter_size_mask):

    # form it into float64
    mask = mask.astype(np.float64)

    # get the pyramids
    lpyr1, filter_vec = build_laplacian_pyramid(im1, mask_levels, filter_size_im)
    lpyr2, filter_vec = build_laplacian_pyramid(im2, mask_levels, filter_size_im)
    gpyr_m, filter_vec_m = build_gaussian_pyramid(mask, mask_levels, filter_size_mask)

    lpyr_blend = []
    for i in range(mask_levels):
        lpyr_blend.append(gpyr_m[i]*lpyr1[i]+(1-gpyr_m[i])*lpyr2[i])

    return (laplacian_to_image(lpyr_blend,filter_vec, [1 for i in range(len(lpyr_blend))])/255).clip(0,1)


def blending_example1():
    im1 = read_image(relpath('mask/im1.jpg'))
    im2 = read_image(relpath('mask/im2.jpg'))
    mask = read_image(relpath('mask/mask.jpg'))

    R = pyramid_blending(im1[::0], im2[::0], mask, mask_levels, filter_size_im, filter_size_mask)
    G = pyramid_blending(im1[::1], im2[::1], mask, mask_levels, filter_size_im, filter_size_mask)
    B = pyramid_blending(im1[::2], im2[::2], mask, mask_levels, filter_size_im, filter_size_mask)

    return [im1, im2, mask, np.dstack((R,G,B))]