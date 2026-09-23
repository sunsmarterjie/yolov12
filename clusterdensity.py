import cv2 as cv
import numpy as np

img = cv.imread("imagewithbounds.jpg")

original = cv.imread("Screenshot 2025-05-19 152029.png")

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

img_erosion = cv.erode(img, kernel, iterations=1)

#Define structuring element.
boundary = img - img_erosion

img_dilate = cv.dilate(boundary, kernel, iterations = 2)

img_blur  = cv.GaussianBlur(img,(5,5),10)

img_enhance = cv.Laplacian(boundary, cv.CV_64F, ksize=1)

# Create a mask for the color
mask = cv.inRange(img_dilate, (0,0,0), (0,255,0))

# Apply the mask to the original image
result = cv.bitwise_and(img,img, mask=mask)

result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

result = cv.bitwise_and(original,original, mask=result)

cv.imshow('result', result)

#components = cv.connectedComponentsWithStats(result)
#(totalcomponents,ID, pixelvalues, centroid) = components

#print(totalcomponents)
cv.waitKey(0)
cv.destroyAllWindows()
