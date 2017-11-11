from sol2 import *

#--------- TEST DFT & FFT ----------#
# print(DFT(np.arange(3)))
# print(IDFT(np.arange(3)))
# print("\n\n")
# print(np.fft.fft(np.arange(3)))
# print(np.fft.ifft(np.arange(3)))
#
# check that matrix of signals is working
# x = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(DFT(x))
# print(np.fft.fft(x))
#
# print(IDFT(x))
# print(np.fft.ifft(x))
######################################

# --------- 2D FOURIER CHECK --------------
# x = np.array([[1,2,3],[4,5,6],[7,8,9]])
#
# print(IDFT2(x))
# print("\n")
# print(np.fft.ifft2(x))


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