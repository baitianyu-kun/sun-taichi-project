import cv2 as cv

img=cv.imread('../imgs/test_pics/VCG211336925751.jpg')
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()