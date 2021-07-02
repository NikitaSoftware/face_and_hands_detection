# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:55:36 2021

@author: Никита
"""

#face detection
#firstly we need to read video

import cv2
import numpy as np

#reading cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "Haarcascades/haarcascade_frontalface_tree_alt.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascades/haarcascade_eye.xml')
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascades/hand.xml')
palm_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascades/palm.xml')

#reading video
video = cv2.VideoCapture("C:\WIN_20210701_11_28_22_Pro.mp4")

#detection

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 0.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew-(x+w), ey+eh-(y+h)), (0, 255, 0), 2)
    hands = hand_cascade.detectMultiScale(gray,2,5)
    for ( hx,hy,hw,hh) in hands:
        cv2.rectangle(frame, (hx,hy), (hx+hw,hy+hh), (255, 0, 0), 2)
        roi_gray = gray[hy:hy+hh, hx:hx + hw]
        roi_color = frame[hy:hy+hh, hx:hx + hw]
        palm = palm_cascade.detectMultiScale(frame, (hx,hy),(hx + hw,hy+hh),(255, 0, 0), 2)
        for ( phx,phy,phw,phh) in palm:
            cv2.rectangle(roi_color, (phx,phy), (phx+phw,phy+phh), (255, 0, 0), 2)
    return frame

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()


