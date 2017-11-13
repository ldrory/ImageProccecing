__author__ = "Liran Drory"

import numpy as np
from scipy.signal import convolve2d as conv
import matplotlib.pylab as plt
from scipy.misc import imread as imread

#TODO: #check the complex128 or the float64 demant on DFT & IDFT


def DFT(signal):

    # find the length of the signal
    N = signal.shape[0]
    if signal.ndim == 2:
        M, N = signal.shape

    # calculate DFT matrix
    u, v  = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2 * np.pi * 1j / N)
    dft_matrix = np.power(omega, u*v)

    # if it is a matrix of signals
    if signal.ndim == 2:
        # calculate the Fourier Transform
        complex_fourier_signal = np.dot(dft_matrix, signal.transpose())
        return complex_fourier_signal.transpose()

    # calculate the Fourier Transform
    complex_fourier_signal = np.dot(dft_matrix, signal)
    return complex_fourier_signal


def IDFT(fourier_signal):

    # find the length of the signal
    N = fourier_signal.shape[0]
    if fourier_signal.ndim == 2:
        M, N = fourier_signal.shape

    # calculate IDFT matrix
    u, v  = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(2 * np.pi * 1j / N)
    idft_matrix = np.power(omega, u*v)

    # if it is a matrix of fourier signals
    if fourier_signal.ndim == 2:
        # calculate the Fourier Transform
        signal = np.dot(idft_matrix, fourier_signal.transpose())
        return 1/N * signal.transpose()

    # calculate the inverse Fourier Transform
    signal = np.dot(idft_matrix, fourier_signal)
    return 1/N * signal


def DFT2(image):
    M, N = image.shape

    # build the dft2_matrix transform
    omega_y = np.exp(-2 * np.pi * 1j / M)
    u, v = np.meshgrid(np.arange(M), np.arange(M))
    dft2_matrix = np.power(omega_y, u*v)

    # calculate the 2D fourier transform
    fourier_image = np.dot(dft2_matrix, DFT(image))

    return fourier_image


def IDFT2(fourier_image):
    M, N = fourier_image.shape
    # build the idft2_matrix transform
    omega_y = np.exp(2 * np.pi * 1j / M)
    u, v = np.meshgrid(np.arange(M), np.arange(M))
    idft2_matrix = np.power(omega_y, u*v)

    # calculate the 2D inverse fourier transform
    return 1/M * np.dot(idft2_matrix, IDFT(fourier_image))


def conv_der(im):
    # set der x/y matrix
    der_x = np.array([[1, 0, -1]])
    der_y = np.array(der_x.transpose())
    # calculate the derivative to x and y
    dx = conv(im, der_x, mode='same')
    dy = conv(im, der_y, mode='same')

    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)  # = magnitude


#TODO: #fix fourier_der func
def fourier_der(im):
    # constants
    M, N = im.shape
    u, v = np.meshgrid(np.arange(N), np.arange(M))[0], np.meshgrid(np.arange(N), np.arange(M))[1]
    u_der, v_der = (2 * np.pi * 1j / N), (2 * np.pi * 1j / M)

    # calculate dx, dy
    dft_shifted = np.fft.fftshift(DFT2(im))

    dx = u_der * IDFT2(np.fft.ifftshift(np.fft.fftshift(u) * dft_shifted))
    dy = v_der * IDFT2(np.fft.ifftshift(np.fft.fftshift(v) * dft_shifted))

    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)  # = magnitude


def gaussian_kernel_factory(kernel_size):

    gaussian = binomial_ker = np.array([[1, 1]])
    while gaussian.shape[1] < kernel_size: gaussian = conv(gaussian, binomial_ker)
    gaussian_kernel = np.ones((kernel_size, kernel_size)) * gaussian * gaussian.transpose()

    return 1 / gaussian_kernel.sum() * gaussian_kernel


def blur_spatial(im, kernel_size):
    # im_padding = np.pad(im, kernel_size, mode='edge')
    return conv(im, gaussian_kernel_factory(kernel_size))


def blur_fourier(im, kernel_size):
    dft = DFT2(im)
    gaus = gaussian_kernel_factory(kernel_size)
    p = np.zeros_like(im)

    nb = im.shape[0]
    na = gaus.shape[0]
    lower = (nb) // 2 - (na // 2)
    upper = (nb // 2) + (na // 2)
    p[lower:upper, lower:upper] = gaus

    dft_gau = DFT2(np.fft.ifftshift(p))
    blur_fourier2 = IDFT2(dft * dft_gau)

    return blur_fourier2