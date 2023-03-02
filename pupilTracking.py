import numpy as np
import cv2
import json

# Load the Haar cascade files for eye detection
eye_cascade = cv2.CascadeClassifier('eye2.xml')

# Start the video stream
capture = cv2.VideoCapture(0)

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

        find_pupil(rightEyeImage, frame)
        find_pupil(leftEyeImage, frame)

    # Display the resulting frame
    cv2.imshow('Pupil Tracking', frame)
    cv2.imshow('Right Eye', rightEyeImage)
    cv2.imshow('Left Eye', leftEyeImage)

    # Stop the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
capture.release()
cv2.destroyAllWindows()