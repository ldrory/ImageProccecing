__author__ = "Liran Drory"

import numpy as np
import matplotlib.pylab as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray

#TODO: #check the complex128 or the float64 demant on DFT & IDFT


def DFT(signal):

    # find the length of the signal
    N = signal.shape[0]

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

    # calculate IDFT matrix
    u, v  = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(2 * np.pi * 1j / N)
    idft_matrix = 1/N * np.power(omega, u*v)

    # if it is a matrix of fourier signals
    if fourier_signal.ndim == 2:
        # calculate the Fourier Transform
        signal = np.dot(idft_matrix, fourier_signal.transpose())
        return signal.transpose()

    # calculate the inverse Fourier Transform
    signal = np.dot(idft_matrix, fourier_signal)
    return signal


def DFT2(image):
    M, N = image.shape

    # build the dft2_matrix transform
    omega_y = np.exp(-2 * np.pi * 1j / M)
    u, v = np.meshgrid(np.arange(M), np.arange(M))
    dft2_matrix = np.power(omega_y, u*v)

    # calculate the 2D fourier transform
    DFT_1D = DFT(image)
    fourier_image = np.dot(dft2_matrix, DFT_1D)

    return fourier_image


def IDFT2(fourier_image):
    M, N = fourier_image.shape

    # build the idft2_matrix transform
    omega_y = np.exp(2 * np.pi * 1j / M)
    u, v = np.meshgrid(np.arange(M), np.arange(M))
    idft2_matrix = np.power(omega_y, u*v)

    # calculate the 2D inverse fourier transform
    IDFT_1D = IDFT(fourier_image)
    image = 1/N * np.dot(idft2_matrix, IDFT_1D)

    return image



