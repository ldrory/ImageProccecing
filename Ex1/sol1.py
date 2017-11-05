import numpy as np
import matplotlib.pylab as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray

GRAYSCAL = 1
GRAYSCAL_MATRIX = 2
RGB = 2
RGB_MATRIX = 3
HIEGTH = 0
WIDTH = 1
SHADES_OF_GRAY = 255
COLORFULL = 3


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
        im_float /= 255                     # pixels [0,1]
        return im_float

    if representation == GRAYSCAL:
        im_g = rgb2gray(im)                 # turn to grey
        return im_g

def imdisplay(filename, representation):
    """
    Reads image and plot it as GRAY or RGB
    depends on the color

    Parameters
    ----------
    :param filename: str
        string containing the image filename to read (PATH)

    :param representation: int
        either 1 or 2 defining whether the output
        should be a greyscale image (1) or an RGB image (2).

    Returns
    -------
    :return void

    """

    # read image to im matrix
    im = read_image(filename, representation)

    # if image is gray than show on intensity
    if len(im.shape) == GRAYSCAL_MATRIX:
        plt.imshow(im, cmap=plt.cm.gray)  # present
        plt.show()

    # if image is RGB than show on rgb
    if len(im.shape) == RGB_MATRIX:
        plt.imshow(im)                    # present
        plt.show()

def rgb2yiq(imRGB):
    """
    transform RGB image to YIQ by constant matrix

    Parameters
    ----------
    :param imRGB: numpy.array
        3D pixels matrix of the imRGB

    Returns
    -------
    :return imYIQ: numpy.array
        3D matrix of YIQ image

    """

    # define the rgb2yiq matrix
    rgb2yiq_matrix = np.array([[0.299, 0.578, 0.144],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
    # separate the RGB matrix to R G & B
    R = np.array(imRGB)[:, :, 0]
    G = np.array(imRGB)[:, :, 1]
    B = np.array(imRGB)[:, :, 2]

    # matrix calculation
    Y = (np.array(rgb2yiq_matrix[0][0] * R + rgb2yiq_matrix[0][1] * G + rgb2yiq_matrix[0][2] * B))
    I = (np.array(rgb2yiq_matrix[1][0] * R + rgb2yiq_matrix[1][1] * G + rgb2yiq_matrix[1][2] * B))
    Q = (np.array(rgb2yiq_matrix[2][0] * R + rgb2yiq_matrix[2][1] * G + rgb2yiq_matrix[2][2] * B))

    # return the imYIQ
    return np.dstack((Y, I, Q))

def yiq2rgb(imYIQ):
    """
    transform YIQ image to RGB by constant matrix

    Parameters
    ----------
    :param imYIQ: numpy.array
        3D pixels matrix of the imYIQ
    Returns
    -------
    :return imRGB: numpy.array
        3D matrix of RGB image

    """

    # define the yiq2rgb matrix
    yiq2rgb_matrix = np.array([ [1,  0.956,  0.621],
                                [1, -0.272, -0.647],
                                [1, -1.106,  1.703]])
    # separate the YIQ matrix to Y I & Q
    Y = np.array(imYIQ)[:, :, 0]
    I = np.array(imYIQ)[:, :, 1]
    Q = np.array(imYIQ)[:, :, 2]

    # matrix calculation
    R = np.array(yiq2rgb_matrix[0][0] * Y + yiq2rgb_matrix[0][1] * I + yiq2rgb_matrix[0][2] * Q)
    G = np.array(yiq2rgb_matrix[1][0] * Y + yiq2rgb_matrix[1][1] * I + yiq2rgb_matrix[1][2] * Q)
    B = np.array(yiq2rgb_matrix[2][0] * Y + yiq2rgb_matrix[2][1] * I + yiq2rgb_matrix[2][2] * Q)

    # return the imYIQ
    return np.dstack((R, G, B))

def histogram_equalize(im_orig):
    """
    this function making histogram equalization on an image

    Parameters
    ----------
    :param im_orig: numpy.array
        3D pixels matrix of the original image

    Returns
    -------
    :return [im_eq, hist_orig, hist_eq]: python list
        im_eq: numpy.array
            3D/2D pixels matrix of the histogram equalized image (RGB/GRAYSCALE)
        hist_orig: numpy.array
            1D histogram of the original image
        hist_eq:
            1D histogram of the equalized image

    """

    # identities
    hieght, width = im_orig.shape[HIEGTH], im_orig.shape[WIDTH]
    n_pixels = hieght*width
    originaly_colorful = False
    im = im_orig

    # check if picture is rgb or gray
    if len(im_orig.shape) == COLORFULL:
        originaly_colorful = True
        im_yiq = rgb2yiq(im_orig)
        Y  = im_yiq[:, :, 0]
        I  = im_yiq[:, :, 1]
        Q  = im_yiq[:, :, 2]
        im = Y

    # calculate histogram and cumulative function
    histogram, bins = np.histogram(im*SHADES_OF_GRAY, bins=np.arange(SHADES_OF_GRAY+2))
    c_histogram = np.cumsum(histogram)

    # normalized cum func
    lut = ((c_histogram/n_pixels)*SHADES_OF_GRAY)  # transform function
    lut = lut.round()

    # check that mim value is 0 and max is K-1 (255)
    # otherwise stretch the result linearly
    if lut[0] != 0 or lut[255] != 255:
        m = np.nonzero(lut)[0][0]
        lut = ((c_histogram - c_histogram[m])/(c_histogram[SHADES_OF_GRAY]-c_histogram[m]))*SHADES_OF_GRAY
        lut = lut.round()

    # pixels to float
    im_eq = (lut[(im*255).astype(int)]).astype(np.float64)/255

    # histogram of original & new image
    hist_orig, bounds_orig = np.histogram(im_orig*SHADES_OF_GRAY, np.arange(257))
    hist_eq, bounds_eq = np.histogram(im_eq*SHADES_OF_GRAY, np.arange(257))

    if originaly_colorful:
        im_eq = yiq2rgb(np.dstack((im_eq, I, Q)))
        im_eq.clip(0, 1, im_eq)

    return [im_eq, hist_orig, hist_eq]

def quantize(im_orig, n_quant, n_iter):
    """
    this function quantize the image

    Parameters
    ----------
    :param im_orig: numpy.array
        3D pixels matrix of the original image
    :param n_quant: int
        number of the quantization
    :param n_iter: int
        number of iteration of the minError loop

    Returns
    -------
    :return [im_quant, error]: python list
        im_quant: numpy.array
            3D/2D pixels matrix of the quantize image (RGB/GRAYSCALE)
        error: numpy.array
            1D histogram of the errors

    """

    # identities
    hieght, width = im_orig.shape[HIEGTH], im_orig.shape[WIDTH]
    n_pixels = hieght*width
    originaly_colorful = False
    im = im_orig

    # check if picture is rgb or gray
    if len(im_orig.shape) == COLORFULL:
        originaly_colorful = True
        im_yiq = rgb2yiq(im_orig)
        Y  = im_yiq[:, :, 0]
        I  = im_yiq[:, :, 1]
        Q  = im_yiq[:, :, 2]
        im = Y.clip(0,1)

    # calculate histogram and cumulative function
    h, bins = np.histogram((im*SHADES_OF_GRAY).astype(int), bins=np.arange(SHADES_OF_GRAY+2))

    # initialized of z and q
    z = np.linspace(0, 255, n_quant+1)
    q = np.zeros(n_quant)
    Enew = float('inf')
    error = np.array([])

    # divide the z smartly (by the number of pixels)
    comulative_hist = h.cumsum()
    eachZone = n_pixels/n_quant
    for i in range(n_quant+1):
        if (i != 0 and i != n_quant):
            equal = np.where(comulative_hist >= eachZone*i)
            if (z[i-1] != z[i]):
                z[i] = equal[0][0]
            # if it's to tight for a z'th move it
            else:
                z[i] = equal[0][0]+1

    # find the qs and the zs and the min error
    # find untill z is stable or n_iter is over
    while(n_iter):
        n_iter -= 1
        min_z = z.copy()
        min_q = q.copy()
        Eold = Enew

        for i in range(n_quant):
            Zsection = (np.arange(z[i], z[i + 1]+1)).astype(int)
            q_up = np.sum(np.multiply(Zsection, h[Zsection]))
            q_down = np.sum(h[Zsection])
            q[i] = q_up // q_down

            if (i==0 or i==n_quant):
                continue
            else:
                z[i] = (q[i-1]+q[i])//2

        Enew = []
        for i in range(n_quant):
            for zk in range(int(z[i]), int(z[i+1])+1):
                Enew.append(pow(q[i]-zk, 2)*h[zk])

        Enew = sum(Enew)
        error = np.append(error, Enew)

        if Eold <= Enew:
            break

    # update the look up table by the q's
    lut = np.zeros(256)
    z = min_z.astype(int)
    q = min_q.astype(int)
    for i in range(n_quant):
        lut[z[i]:z[i+1]] += q[i]

    # update the last one
    lut[255] = q[n_quant-1]

    # every pixel is transfer according to the lut
    im_quant = (lut[(im*255).astype(int)]).astype(np.float64)/255

    if originaly_colorful:
        im_quant = yiq2rgb(np.dstack((im_quant, I, Q)))
        im_quant.clip(0, 1, im_quant)

    return [im_quant, error]

