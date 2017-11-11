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

    # calculate the Fourier Transform
    signal = np.dot(idft_matrix, fourier_signal)

    return signal


