import cv2 as cv

img = cv.imread('../data/Resources/Photos/cat_large.jpg')

def RescaleImage(frame,scale=0.75):
    width  = int(frame[1]*scale)
    height  = (frame[0]*scale)
    dimensions = (width,height)


    return cv.resize(framedimensions,interpolation = cv.INTER_AREA)

capture =cv.VideoCapture('../data/Resources/Videos/dog.mp4')

while True:
    isTrue,frame =capture.read()
    
