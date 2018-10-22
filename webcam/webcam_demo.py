import numpy as np
import cv2
import dlib
import openface
import pickle

from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time

print("Loading models...")
predictor_model = "shape_predictor_68_face_landmarks.dat"
align = openface.AlignDlib(predictor_model)
net = openface.TorchNeuralNet("nn4.small2.v1.t7")
with open('classifier.pkl', 'rb') as f:
    (le, clf) = pickle.load(f)
face_detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("Camera sensor warming up...")
vs = VideoStream(usePiCamera=0).start()
time.sleep(2.0)

while(True):
    # Capture frame-by-frame
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detected_faces = face_detector(gray, 0)
    # detected_faces = face_cascade.detectMultiScale(gray)
    bbs = align.getAllFaceBoundingBoxes(frame)

    for i, face_rect in enumerate(bbs):
        cv2.rectangle(frame, (face_rect.left(),face_rect.top()),(face_rect.right(),face_rect.bottom()),
                    (0, 255, 0),2)

        alignedFace = align.align(96, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        feature = net.forward(alignedFace)
        feature = feature.reshape(1, -1)

        predictions = clf.predict_proba(feature).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

        if confidence < 0.3:
            cv2.putText(frame, 'Unknown', (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
        else:
            cv2.putText(frame, person, (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
vs.stop()
