import cv2
import numpy as np

def detect_pupil(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX,cY)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    blur = cv2.GaussianBlur(dst,(5,5),0)
    inv = cv2.bitwise_not(blur)
    thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    ret,thresh1 = cv2.threshold(erosion,210,255,cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = 10000
    final_cnt = None
    for cnt in cnts:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        distance = abs(centroid[0]-x)+abs(centroid[1]-y)
        if distance < flag :
            flag = distance
            final_cnt = cnt
        else:
            continue
    (x,y),radius = cv2.minEnclosingCircle(final_cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(img,center,radius,(255,0,0),2)

    return img

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        detected_pupil = detect_pupil(frame)
        cv2.imshow("Result", detected_pupil)
    else:
        print("Error capturing frames")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()