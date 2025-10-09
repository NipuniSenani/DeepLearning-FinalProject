# import the necessary packages
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import cv2

import matplotlib.pyplot as plt

from keras.models import load_model

eye_model = load_model('/Users/nipuni/Desktop/CS 570/Final_Project/bestModel.h5')

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
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Set the desired frame rate
frame_rate = 1  # 1 frame per second

# Initialize the time of the last captured frame
last_frame_time = time.time()

drowsy = False
close_eye_cont = 0
consecative_frames = 0
# loop over the frames from the video stream
while True:
    if time.time() - last_frame_time >= 2 / frame_rate:
        # Update the last captured frame time
        last_frame_time = time.time()
        # grab the frame from the video stream, resize it, and convert it
        # to grayscale
        frame = vs.read()

        if consecative_frames == 1:
            consecative_frames += 1

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # perform face detection using the appropriate haar cascade
        faceRects = detectors["face"].detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # loop over the face bounding boxes
        for (fX, fY, fW, fH) in faceRects:
            # extract the face ROI
            faceROI = gray[fY:fY + fH, fX:fX + fW]
            # apply eyes detection to the face ROI
            eyeRects = detectors["eyes"].detectMultiScale(
                faceROI, scaleFactor=1.1, minNeighbors=10,
                minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            # extract the eye ROI
            for (eX, eY, eW, eH) in eyeRects:
                eyeROI = faceROI[eY:eY + eH, eX:eX + eW]

                plt.imshow(eyeROI)
                plt.show()
                # Resize the eyeROI to the desired shape (24, 24)
                eyeROI = cv2.resize(eyeROI, (24, 24))
                eyeROI = np.expand_dims(eyeROI, -1)

                if eye_model.predict(np.expand_dims(eyeROI, 0)) < 0.5:
                    close_eye_cont += 1
                    consecative_frames += 1

                    if close_eye_cont > 2 and consecative_frames > 1:
                        print('Driver is drowsy')
                        close_eye_cont = 0
                        consecative_frames = 0


            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
