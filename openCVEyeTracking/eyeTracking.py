import numpy as np
import cv2
import json

# Load the Haar cascade files for face detection
eye_cascade = cv2.CascadeClassifier('eye2.xml')
face_cascade = cv2.CascadeClassifier('face.xml')

# Start the video stream
capture = cv2.VideoCapture(0)

class iris_detection():
    def __init__(self, _img):
        '''
        initialize the class and set the class attributes
        '''
        self._img = _img
        self._pupil = None
        
    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def detect_pupil(self):
        '''
        This method should use cv2.findContours and cv2.HoughCircles() function from cv2 library to find the pupil
        and then set the coordinates for pupil circle coordinates
        '''
        # First binarize the image so that findContours can work correctly.
        _, thresh = cv2.threshold(self._img, 100, 255, cv2.THRESH_BINARY)
        # Now find the contours and then find the pupil in the contours.
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, self._img.shape[0]/2)
        # Then mask the pupil from the image and store it's coordinates.
        for l in c:
            # OpenCV returns the circles as a list of lists of circles
            for circle in l:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(self._img, center, radius, (0, 0, 0), thickness = -1)
                self._pupil = (center[0], center[1], radius)
        
    def detect_iris(self):
        '''
        This method should use the background subtraction technique to isolate the iris from the original image
        It should use the coordinates from the detect_pupil to get a larger circle using cv2.HoughCircles()
        '''
        _, t = cv2.threshold(self._img, 195, 255, cv2.THRESH_BINARY)
        contours, _, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the iris using the radius of the pupil as input.
        c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 2, param2 = 150)

        for l in c:
            for circle in l:
                center = (self._pupil[0], self._pupil[1])
                radius = circle[2]
                # This creates a black image and draws an iris-sized white circle in it.
                mask = numpy.zeros((self._img.shape[0], self._img.shape[1], 1), numpy.uint8)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness = -1)
                # Mask the iris and crop everything outside of its radius.
                self._img = cv2.bitwise_and(self._img, mask)

    def start_detection(self):
        self.convert_to_gray_scale()
        self.detect_pupil()
        self.detect_iris()
        return self._img

def detect_pupil(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX,cY)
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    #blur = cv2.GaussianBlur(dst,(5,5),0)
    #inv = cv2.bitwise_not(blur)
    #thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((2,2),np.uint8)
    #erosion = cv2.erode(thresh,kernel,iterations = 1)
    #ret,thresh1 = cv2.threshold(erosion,210,255,cv2.THRESH_BINARY)
    
    # Threshold the image to exclude black pixels
    #ret, black_mask = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = 10000
    final_cnt = None
    height, width, channel = img.shape
    print("<--------->")
    for cnt in contours:
        print("------")
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        distance = abs(centroid[0]-x) + abs(centroid[1]-y)
        if distance < flag and distance < 10:
            flag = distance
            final_cnt = cnt
        print(distance)
            
    try:
        (x,y),radius = cv2.minEnclosingCircle(final_cnt)
        center = (int(x),int(y))
        radius = int(radius)
        img = cv2.circle(img,center,radius,(255,0,0),2)
    except:
        print("couldn't find pupils")

    return img

def draw_pupil_square(eye_img):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold to the grayscale image
    _, threshold = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Find the contours in the threshold image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the pupil
    (x, y, w, h) = cv2.boundingRect(max_contour)

    # Crop the region of interest around the pupil
    roi = eye_img[y:y+h, x:x+w]

    # Convert the cropped region to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, roi_threshold = cv2.threshold(roi_gray, 60, 255, cv2.THRESH_BINARY)

    # Find the contours in the threshold image
    roi_contours, _ = cv2.findContours(roi_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area
    max_roi_contour = max(roi_contours, key=cv2.contourArea)

    # Get the bounding rectangle of the pupil in the region of interest
    (roi_x, roi_y, roi_w, roi_h) = cv2.boundingRect(max_roi_contour)

    # Draw a square around the pupil
    cv2.rectangle(eye_img, (x + roi_x, y + roi_y), (x + roi_x + roi_w, y + roi_y + roi_h), (0, 255, 0), 2)

    # Return the image with the square drawn around the pupil
    return eye_img

def circle_brightest_area(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to the grayscale image to convert it to binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    max_area = 0
    brightest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            brightest_contour = contour
    
    # Calculate the center and radius of the circle that fits the contour
    (x, y), radius = cv2.minEnclosingCircle(brightest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Draw the circle on the original image
    cv2.circle(image, center, radius, (0, 255, 0), 2)

    return image

def detect_blue_pixels(image, circleFrame):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of blue pixels in the HSV color space
    lower_blue = (155, 54, 100)
    upper_blue = (225, 100, 100)

    # Create a mask of blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find the contours of the blue pixels
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the center of all blue pixels
    moments = cv2.moments(mask)
    if(moments["m00"] != 0):
        center_of_mass = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        # Draw a circle around all blue pixels
        cv2.circle(circleFrame, center_of_mass, int(np.sqrt(moments["m00"]/np.pi)), (0, 255, 0), 2)
    else:
        text('failed to find pupils', (10, 500), 1, (255, 255, 255))

def detect_pupils(img):
    # Load image
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to the image to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Use the Hough Circle Transform to detect circles in the image
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    # check if any pupils are detected
    if circles is not None:
        pupils = []
        for circle in circles[0, :]:
            x, y, r = circle
            pupils.append((x, y, r))
        return pupils
    else:
        return None

def find_pupil(img, draw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the HoughCircles function to detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # Make sure at least one circle was found
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the circles
        for (x, y, r) in circles:
            # Draw the circle in the output image
            #cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            # Draw a rectangle as the pupil
            cv2.rectangle(draw, (x-r, y-r), (x+r, y+r), (255, 0, 0), 2)
    else:
        text('Failed to find pupils', (10, 500), 1, (255, 255, 255))


def text(text,bottomLeftCornerOfText,fontScale,fontColor):

    cv2.putText(frame,text, 
        bottomLeftCornerOfText, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale,
        fontColor,
        1, #thickness
        2) #linetype

def create_polygon_image(points, frame):
    # Create a blank image with a black background
    img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

    # Convert the points to a format that can be used by fillConvexPoly
    points = np.array(points, np.int32)
    points = points.reshape((-1,1,2))

    # Use the fillConvexPoly function to create the polygon
    cv2.fillConvexPoly(img, points, (255,255,255))

    img = img * frame

    # Get the bounding rect of the polygon
    x, y, w, h = cv2.boundingRect(points)

    # Crop the image to the bounding rect of the polygon
    img = img[y + 2:y+h - 2, x + 2:x+w - 2]

    return img

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    if(len(eyes) > 1):
        # Draw a rectangle around the eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if(eyes[0][0] < eyes[1][0]): #if eye2 is further right than eye1
            eye1 = eyes[0]
            eye2 = eyes[1]
        else:
            eye1 = eyes[1]
            eye2 = eyes[0]

        #text("eye1: %s" % (eye1,), (10, 500), 1, (255, 255, 255))

        eye1Polygon = [ #
            (eye1[0], eye1[1]), #x, y
            (eye1[0] + eye1[2], eye1[1]), #x + w, y
            (eye1[0] + eye1[2], eye1[1] + eye1[3]), #x + w, y + h
            (eye1[0], eye1[1] + eye1[3]) #x, y + h
        ]

        eye2Polygon = [
            (eye2[0], eye2[1]), #x, y
            (eye2[0] + eye2[2], eye2[1]), #x + w, y
            (eye2[0] + eye2[2], eye2[1] + eye2[3]), #x + w, y + h
            (eye2[0], eye2[1] + eye2[3]) #x, y + h
        ]

        rightEyeImage = create_polygon_image(eye1Polygon, frame)
        leftEyeImage = create_polygon_image(eye2Polygon, frame)

        #find_pupil(rightEyeImage, frame)
        #find_pupil(leftEyeImage, frame)

        #eyeFrame = cv2.add(rightEyeImage, leftEyeImage)

        #detect_blue_pixels(eyeFrame, frame)

        #leftPupil = detect_pupils(leftEyeImage)
        #rightPupil = detect_pupils(rightEyeImage)

        #text(leftPupil, (10, 500), 1, (255, 255, 255))
        #text(leftPupil, (10, 600), 1, (255, 255, 255))

        leftPupil = detect_pupil(leftEyeImage)
        rightPupil = detect_pupil(rightEyeImage)

        #leftPupilClass = iris_detection(leftEyeImage)
        #rightPupilClass = iris_detection(rightEyeImage)

        #leftPupil = leftPupilClass.start_detection()
        #rightPupil = rightPupilClass.start_detection()

    # Display the resulting frame
    cv2.imshow('Pupil Tracking', frame)
    #cv2.imshow('Eye Detection', eyeFrame)
    cv2.imshow('Right Eye', rightEyeImage)
    cv2.imshow('Left Eye', leftEyeImage)
    cv2.imshow('Right Eye', rightPupil)
    cv2.imshow('Left Eye', leftPupil)

    # Stop the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
capture.release()
cv2.destroyAllWindows()