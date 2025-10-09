# import the necessary packages
import imageio
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import cv2

import matplotlib.pyplot as plt
from keras.models import load_model

import dlib

eye_model = load_model('/Users/nipuni/Desktop/CS 570/Final Project/Final_Project/bestModel.keras')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0, framerate=25).start()
time.sleep(2.0)

eye_flip_count = 0
eye_open = False
eye_close = False
eye_frame_count = 0

# Set the start time
start_time = time.time()
reset_interval = 10  # in seconds

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it, and convert it
    # to grayscale
    frame = vs.read()
    eye_frame_count += 1
    # Check if 1 minute has passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= reset_interval:
        print(f"{reset_interval} seconds has passed: \n eye flips :{eye_flip_count}, no of frames captured :{eye_frame_count}")
        if eye_flip_count < 4:
            print("Driver is Drowsy.  Consider Taking A Break")
        # Reset eye flip
        eye_flip_count = 0
        # Update the start time for the next interval
        start_time = time.time()
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform face detection using the appropriate haar cascade
    faceRects = face_detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # the face bounding boxes
    if len(faceRects) > 0:
        (fX, fY, fW, fH) = faceRects[0]
    else:
        continue

    # extract the face ROI
    faceROI = gray[fY:fY + fH, fX:fX + fW]
    # faceROI = gray
    # Detect faces in the faceROI
    detector = dlib.get_frontal_face_detector()
    detect = detector(faceROI, 1)
    if len(detect) > 0 :
        # Landmarks
        shape = predictor(faceROI, detect[0])
    else:
        continue

    # Left Eyes is 37 to 40 and 39 to 42
    x1 = shape.part(36).x
    x2 = shape.part(39).x
    y1 = shape.part(42).y
    y2 = shape.part(45).y

    left_eye = faceROI[y1 - 10:y2 + 10, x1-10:x2+10]

    # Right Eyes is 43 to 46 and 44 to 47.
    x1 = shape.part(42).x
    x2 = shape.part(45).x
    y1 = shape.part(44).y
    y2 = shape.part(47).y

    right_eye = faceROI[y1 - 8:y2 + 8, x1-4:x2+4]

    # Resize the eye regions to 24x24
    left_eye = np.expand_dims(cv2.resize(left_eye, (24, 24)), -1)
    right_eye = np.expand_dims(cv2.resize(right_eye, (24, 24)), -1)

    # Predicting whether eye are close or open
    left_eye_close = eye_model.predict(np.expand_dims(left_eye, 0))

    print(f"CNN model predicting value : {left_eye_close[0][0]}")

    if left_eye_close > 0.5:
        eye_close = True
        plt.subplot(1, 2, 1)
        plt.imshow(left_eye)
        plt.subplot(1, 2, 2)
        plt.imshow(right_eye)
        plt.show()
    else:
        eye_open = True

    # Logic for eye flip count
    if eye_open:
        if eye_close:
            eye_flip_count += 1
            eye_close = False
            eye_open = False

    if eye_close:
        if eye_open:
            eye_flip_count += 1
            eye_close = False
            eye_open = False

    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),(0, 255, 0), 2)
    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
