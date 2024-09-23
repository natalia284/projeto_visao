import cv2
from seam import SeamCarve

img = cv2.imread('/home/natalia/Música/praticas_visão/pratica_03_seam_remove/Example/test.png')
mask = cv2.imread('/home/natalia/Música/praticas_visão/pratica_03_seam_remove/Example/mascara.png', 0) != 255

sc_img = SeamCarve(img)
sc_img.remove_mask(mask)

cv2.imshow('original', img)
cv2.imshow('removed', sc_img.image())
cv2.waitKey(0)