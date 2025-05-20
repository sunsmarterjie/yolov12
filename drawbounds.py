#program to manually draw boxes and find the amount of boxes within those bounds
import cv2 as cv

#manually draw circles on images
img = cv.imread("Screenshot 2025-05-19 152029.png")
drawing = False
mode = False
ix, iy = -1,-1
def draw_circle(event, x, y, flags, param):
    global drawing,mode,ix,iy
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv.rectangle(img, (ix,iy), (x, y), (0,255,0), -1)
            else:
                cv.circle(img, (x, y), 25, (0, 255, 0), -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 25, (0, 255, 0), -1)



cv.namedWindow(winname="test drawing")
cv.setMouseCallback("test drawing", draw_circle)
while True:
    cv.imshow('test drawing',img)
    k = cv.waitKey(1)
    if k == ord('m'):
        mode = not mode
    elif k == ord('s'):
        cv.imwrite("imagewithbounds.jpg", img)

#save new image



cv.destroyAllWindows()
