import numpy as np
import matplotlib.pylab as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import math

#np.set_printoptions(threshold=np.inf)

GRAYSCAL = 1
GRAYSCAL_MATRIX = 2
RGB = 2
RGB_MATRIX = 3
HIEGTH = 0
WIDTH = 1
SHADES_OF_GRAY = 255

# filename - string containing the image filename to read.
# representation - representation code,
# either 1 or 2 defining whether the output
# should be a greyscale image (1) or an RGB image (2).
def read_image(filename, representation):

    # load the image
    im = imread(filename)

    if representation == RGB:
        im_float = im.astype(np.float64)    # pixels to float
        im_float /= 255                     # pixels [0,1]
        return im_float

    if representation == GRAYSCAL:
        im_g = rgb2gray(im)                 # turn to grey
        return im_g

def imdisplay(filename, representation):

    # read image to im
    im = read_image(filename, representation)

    # if image is gray than show on intensity
    if len(im.shape) == GRAYSCAL_MATRIX:
        plt.imshow(im, cmap=plt.cm.gray)  # present
        plt.show()

    # if image is RGB than show on rgb
    if len(im.shape) == RGB_MATRIX:
        plt.imshow(im)
        plt.show()

def rgb2yiq(imRGB):

    # define the rgb2yiq matrix
    rgb2yiq_matrix = np.array([[0.299, 0.578, 0.144],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
    # separate the RGB matrix to R G & B
    R = np.array(imRGB)[:, :, 0]
    G = np.array(imRGB)[:, :, 1]
    B = np.array(imRGB)[:, :, 2]

    # matrix calculation
    Y = (np.array(rgb2yiq_matrix[0][0] * R + rgb2yiq_matrix[0][1] * G + rgb2yiq_matrix[0][2] * B)).clip(0,1)
    I = (np.array(rgb2yiq_matrix[1][0] * R + rgb2yiq_matrix[1][1] * G + rgb2yiq_matrix[1][2] * B)).clip(-0.5957,0.5957)
    Q = (np.array(rgb2yiq_matrix[2][0] * R + rgb2yiq_matrix[2][1] * G + rgb2yiq_matrix[2][2] * B)).clip(-0.5226,0.5226)

    # return the imYIQ
    return np.dstack((Y, I, Q))

def yiq2rgb(imYIQ):

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
    return np.dstack((R, G, B)).clip(0,1)

def histogram_equalize(im_orig):          #input: grayscale or RGB with [0,1] values

    # some identities
    hieght, width = im_orig.shape[HIEGTH], im_orig.shape[WIDTH]
    n_pixels = hieght*width
    originaly_colorful = False
    im = im_orig


    # check if picture is rgb or gray
    if len(im_orig.shape) == 3:
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
    sk = ((c_histogram/n_pixels)*SHADES_OF_GRAY)  # transform function
    sk = sk.round()

    # check that mim value is 0 and max is K-1 (255)
    # otherwise stretch the result linearly
    if sk[0] != 0 or sk[255] != 255:
        m = np.nonzero(sk)[0][0]
        sk = ((c_histogram - c_histogram[m])/(c_histogram[255]-c_histogram[m]))*255
        sk = sk.round()


    # pixels to float
    im_eq = (sk[(im*255).astype(int)]).astype(np.float64)/255


    # histogram of original & new image
    hist_orig, bounds_orig = np.histogram(im_orig*SHADES_OF_GRAY, np.arange(257))
    hist_eq, bounds_eq = np.histogram(im_eq*SHADES_OF_GRAY, np.arange(257))

    if originaly_colorful:
        im_eq = yiq2rgb(np.dstack((im_eq, I, Q)))
        im_eq.clip(0,1,im_eq)

    return [im_eq, hist_orig, hist_eq]

def quantize(im_orig, n_quant, n_iter):

    # some identities
    hieght, width = im_orig.shape[HIEGTH], im_orig.shape[WIDTH]
    n_pixels = hieght*width
    originaly_colorful = False
    im = im_orig

    # check if picture is rgb or gray
    if len(im_orig.shape) == 3:
        originaly_colorful = True
        im_yiq = rgb2yiq(im_orig)
        Y  = im_yiq[:, :, 0]
        I  = im_yiq[:, :, 1]
        Q  = im_yiq[:, :, 2]
        im = Y


    # calculate histogram and cumulative function
    h, bins = np.histogram(im*SHADES_OF_GRAY, bins=np.arange(SHADES_OF_GRAY+2))
    min = np.array([])


    #initialized of z and q
    z = np.linspace(0,255,n_quant+1)
    q = np.zeros(n_quant)
    Enew = float('inf')

    for j in range(n_iter):
        min_z = z
        min_q = q
        Eold = Enew

        for i in range(n_quant):
            z[i] = (q[i]+q[i+1])/2
            q_down = np.sum(h[np.arange(z[i], z[i + 1])])
            q_up = np.sum(np.multiply(np.arange(z[i], z[i + 1]),
                                      h[np.arange(z[i], z[i + 1])]))
            q[i] = q_up / q_down

        for i in range(n_quant):
            for z in range(z[i],z[i+1]):
                Enew = ((q[i]-z)^2)*h(z)

        if Eold < Enew:
            break

    lut = np.zeros(256)
    for z in min_z:
        i = 0
        lut = np.append([q[i]]*min_z)
        i += 1

    im_quant = lut[im]





    if originaly_colorful:
        im_quant = yiq2rgb(np.dstack((im_quant, I, Q)))
        im_quant.clip(0,1,im_eq)

    #return [im_quant, error]
    return im_quant
#
#
# image = read_image("C:\\Users\\Liran\\Desktop\\1.jpg",RGB)
#
# im  = quantize(image,3,2)
#
# plt.imshow(image)
# plt.show()
#
# plt.imshow(im)
# plt.show()
#
#
#
#
#




# check HISTOGRAM ###############
image = read_image("C:\\Users\\Liran\\Desktop\\1.jpg",RGB)

im, hist_orig, hist_eq  = histogram_equalize(image)

plt.imshow(image)
plt.show()

plt.imshow(im)
plt.show()

plt.bar(np.arange(256), hist_orig)
plt.show()
plt.bar(np.arange(256), hist_eq)
plt.show()
plt.bar(np.arange(256), np.cumsum(hist_orig))
plt.show()
plt.bar(np.arange(256), np.cumsum(hist_eq))
plt.show()
## check HISTOGRAM ###############
#
























#imcheck = read_image("C:\\Users\\Liran\\Desktop\\lizi.jpg",1)



#hist, bounds = np.histogram(Y, 128)
#plt.tick_params(labelsize=20)
#plt.plot((bounds[:-1] + bounds[1:]) / 2, hist)
#plt.hist(Y.flatten(), bins=128)
#plt.show()


# plt.tick_params(labelsize=20)
# plt.plot((bounds_eq[:-1] + bounds_eq[1:]) / 2, hist_eq)
# plt.hist(im_eq.flatten(), bins=256)
#
# plt.plot((bounds_orig[:-1] + bounds_orig[1:]) / 2, hist_orig)
# plt.hist(im_orig.flatten(), bins=256)
#
# plt.show()

