import numpy as np
import cv2
''' Generate a rotated version of the image to test keyPoint Matching Algo'''
def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

img = cv2.imread('beer_test.jpg')
img_rotate = rotateImage(img,20)
# cv2.imshow('original',img)
# cv2.imshow('rotated',img_rotate)
# cv2.waitKey(0)
cv2.imwrite('rotated_image.jpg',img_rotate)

# cv2.destroyAllWindows()