from sol3 import *

### ------- GAUSSIAN KERNEL CHECK ------- ###
# print(gaussian_kernel(1))
# print(gaussian_kernel(2))
# print(gaussian_kernel(3))
# print(gaussian_kernel(4))
#############################################


### ------ BUILD GAUSSIAN PYRAMID ------- ###
# im = read_image(r'gray_orig.png' ,1)
# pyr, filter_vec = build_gaussian_pyramid(im, 3, 3)
#
# print(len(pyr))
# print(filter_vec, len(filter_vec))
#
# for i in range(len(pyr)):
#     plt.figure()
#     plt.imshow(pyr[i], cmap=plt.cm.gray)
#
# plt.show()
##############################################


### ------ BUILD LAPLACIAN PYRAMID ------- ###
# im = read_image(r'gray_orig.png', 1)
# pyr, filter_vec = build_laplacian_pyramid(im, 4, 3)
#
# print(len(pyr))
# print(filter_vec, len(filter_vec))
#
# for i in range(len(pyr)):
#     plt.figure()
#    # pyr[i] = (pyr[i] - pyr[i].min())/(pyr[i].max()-pyr[i].min())
#     print(pyr[i].max(), pyr[i].min())
#
#     plt.imshow(pyr[i], cmap=plt.cm.gray)
#
# plt.show()
##############################################

### ------ BUILD LAPLACIAN TO IMAGE ------- ###
#
# im = read_image(r'gray_orig.png', 1)
# pyr, filter_vec = build_laplacian_pyramid(im, 4, 3)
#
# plt.figure()
# plt.imshow(im, cmap=plt.cm.gray)
#
#
# img = laplacian_to_image(pyr, filter_vec, [1,1,1,1])
# plt.figure()
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()
################################################

### ------------- render_pyramid --------------###

# im = read_image(r'gray_orig.png', 1)
# pyr, filter_vec = build_gaussian_pyramid(im, 4, 3)
# display_pyramid(pyr,4)
#
# pyr, filter_vec = build_laplacian_pyramid(im, 4, 3)
# display_pyramid(pyr,4)
#

###################################################


### -------------- PYRAMID BLEND ----------------####

im1 = read_image(r'C:\Users\Liran\Documents\GitHub\ImageProccecing\Ex3\mask\im1.png', 1)
im2 = read_image(r'C:\Users\Liran\Documents\GitHub\ImageProccecing\Ex3\mask\im2.png', 1)
mask = read_image(r'C:\Users\Liran\Documents\GitHub\ImageProccecing\Ex3\mask\mask.png', 1)
mask_levels = 5
filter_size_im = 3
filter_size_mask = 3
mask = mask.astype(np.bool)
plt.imshow(mask, cmap=plt.cm.gray)
plt.show()

blended = pyramid_blending(im1, im2, mask, mask_levels, filter_size_im, filter_size_mask)
print(blended.dtype)
plt.imshow(blended, cmap=plt.cm.gray)
plt.show()