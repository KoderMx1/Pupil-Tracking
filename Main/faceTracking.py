import cv2

#load .xml file
eye_cascade = cv2.CascadeClassifier('eye2.xml')

capture = cv2.VideoCapture(0)

#main while loop
while True:

	#read frame
	ret, frame = capture.read()

	#convert the frame to gray scale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #get potential eye locations
	eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

	#go through the first two locations
	for (x, y, w, h) in eyes[:2]:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
	
	#display the image with rectangles
	cv2.imshow('Eye Tracking', frame)
	
	#Exit the while loop if esc is pressed
	key = cv2.waitKey(1)
	if(key == 27): #esc to quit
		#cv2.imwrite('face.jpg', frame)
		break

#release webcam and close the window
capture.release()
cv2.destroyAllWindows()
