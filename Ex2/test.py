from sol2 import *

#--------- TEST DFT & FFT secotion: 1.1----------#
# x = np.arange(3)
# print(DFT(x))
# print(np.fft.fft(x))
#
# print(IDFT(x))
# print(np.fft.ifft(x))
#
# # check that matrix of signals is working
# x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print(DFT(x))
# print(np.fft.fft(x))
#
# print(IDFT(x))
# print(np.fft.ifft(x))
######################################

# --------- 2D FOURIER CHECK secotion: 1.2 --------------
# x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
# x = DFT2(x)
# plt.imshow(x.real)  # present
# plt.show()
# #
# print(IDFT2(x))
# print(np.fft.ifft2(x))
# print("\n")
# print(np.fft.ifft2(np.fft.fft2(x)))
# print(IDFT2(DFT2(x)))
#
#
# #
# k = np.log(1+np.abs(DFT2(x)))
#
# plt.plot(k)
# plt.show()

# ----------------- FOURIER PLOT -------------------------------------
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = DFT(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#
# plt.plot(y)
# plt.grid()
# plt.show()
#
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()
#
# plt.plot(IDFT(DFT(y)).real)
# plt.grid()
# plt.show()

# ----------------- CONV DER secotion: 2.1 -------------------------------------

# im = conv_der(imread(r"grayscale.png", mode='L'))
# plt.figure()
# plt.imshow(im, cmap=plt.cm.gray)  # present
# plt.show()

# ----------------- FOURIER DER secotion: 2.2 -------------------------------------
# #
# im = fourier_der(imread(r"grayscale.png", mode='L'))
# # im = fourier_der(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
# plt.figure()
# plt.imshow(im, cmap=plt.cm.gray)  # present
# plt.show()
#

# ----------------- BLURRING IMAGE secotion: 3.1 -------------------------------------
#
# print(gaussian_kernel_factory(7))
# #
# im = blur_spatial(imread(r"grayscale.png", mode='L'), 25)
# plt.imshow(im, cmap=plt.cm.gray)  # present
# plt.show()

# ----------------- BLURRING IMAGE  FOURIER secotion: 3.2 -------------------------------------
#
# im = blur_fourier(imread(r"grayscale.png", mode='L'), 25)
# plt.imshow(im, cmap=plt.cm.gray)  # present
# plt.show()
#
#
