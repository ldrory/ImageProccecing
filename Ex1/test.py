from sol1 import *

###############CHECK QUANTIZE################################

# image = read_image("C:\\Users\\Liran\\Desktop\\gray_orig.png",RGB)
#
# im, error  = quantize(image, 40, 1000)
#
# plt.imshow(im, cmap=plt.cm.gray)
# plt.show()
#
# print(error)
# plt.plot(error)
# plt.show()

#####################################################


##################### check HISTOGRAM ###############
# image = read_image("C:\\Users\\Liran\\Desktop\\test.png",RGB)
#
# im, hist_orig, hist_eq  = histogram_equalize(image)
#
# plt.subplot(121)
# plt.imshow(image, cmap=plt.cm.gray)
#
# plt.subplot(122)
# plt.imshow(im, cmap=plt.cm.gray)
#
# #plt.subplot_tool()
# plt.show()
#
#
# plt.subplot(221)
# plt.bar(np.arange(256), hist_orig)
#
# plt.subplot(222)
# plt.bar(np.arange(256), hist_eq)
#
# plt.subplot(223)
# plt.bar(np.arange(256), np.cumsum(hist_orig))
#
# plt.subplot(224)
# plt.bar(np.arange(256), np.cumsum(hist_eq))
# plt.show()
## check HISTOGRAM ###############
