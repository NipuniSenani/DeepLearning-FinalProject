# import the necessary packages
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import cv2

import matplotlib.pyplot as plt

from keras.models import load_model

eye_model = load_model('/Users/nipuni/Desktop/CS 570/Final_Project/bestModel.keras')

# initialize a dictionary that maps the name of the haar cascades to
# their filenames
detectorPaths = {
    "face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
}
# initialize a dictionary to store our haar cascade detectors

detectors = {}
# loop over our detector paths
for (name, path) in detectorPaths.items():
    # load the haar cascade from disk and store it in the detectors
    # dictionary
    detectors[name] = cv2.CascadeClassifier(cv2.data.haarcascades + path)

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

    # Check if 1 minute has passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= reset_interval:
        print(f"{reset_interval} seconds has passed: {eye_flip_count},{eye_frame_count}")
        if eye_flip_count < 2:
            print("Driver is Drowsy.  Consider Taking A Break")
        # Reset eye flip
        eye_flip_count = 0
        # Update the start time for the next interval
        start_time = time.time()
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform face detection using the appropriate haar cascade
    faceRects = detectors["face"].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # the face bounding boxes
    if len(faceRects) > 0:
        (fX, fY, fW, fH) = faceRects[0]
    else:
        continue

    # extract the face ROI
    faceROI = gray[fY:fY + fH, fX:fX + fW]
    # apply eyes detection to the face ROI
    eyeRects = detectors["eyes"].detectMultiScale(
        faceROI, scaleFactor=1.001, minNeighbors=5,
        minSize=(7, 7), flags=cv2.CASCADE_SCALE_IMAGE)

    eye_frame_count += len(eyeRects) if len(eyeRects)>0 else 0

    print(f" eye frames {len(eyeRects)}")
    # extract the eye ROI
    for (eX, eY, eW, eH) in eyeRects:
        eyeROI = faceROI[eY:eY + eH, eX:eX + eW]


        # Resize the eyeROI to the desired shape (24, 24)
        eyeROI = cv2.resize(eyeROI, (24, 24))
        eyeROI = np.expand_dims(eyeROI, -1)

        # Predicting whether eye is close or open

        if eye_model.predict(np.expand_dims(eyeROI, 0)) > 0.5:
            eye_close = True
            # print(len(faceRects), len(eyeRects))
            plt.imshow(eyeROI)
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

    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                  (0, 255, 0), 2)
    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
