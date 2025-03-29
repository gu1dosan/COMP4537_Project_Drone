# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import torch
from PIL import Image
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path)

import cv2

import numpy as np

from djitellopy import tello

import time

me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()

me.takeoff()

me.send_rc_control(0, 0, 25, 0)

time.sleep(2.2)

w, h = 360, 240

fbRange = [6200, 6800]

pid = [0.4, 0.4, 0]

pError = 0


def findFace(img):
    # Convert image format
    # pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    pil_img = Image.fromarray(img)
    
    # # Get model predictions
    # with torch.no_grad():
    output = model(pil_img)
        
    results = Detections.from_ultralytics(output[0])
    # print(results)

    # Filter face detections
    boxes = results.xyxy[results.confidence > 0.7]
    
    # Process detected faces
    myFaceListC = []
    myFaceListArea = []
    
    # Process each detection
    for box in boxes:
        # Convert float coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        
        # Calculate center and area
        cx = x1 + w // 2
        cy = y1 + h // 2
        area = w * h
        
        # Draw rectangle (red in RGB)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw center (green in RGB)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        
        # Store face information
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    
    # Return the largest face
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace( info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    #print(speed, fb)
    me.send_rc_control(0, fb, 0, speed)
    return error

#cap = cv2.VideoCapture(1)

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))

    img, info = findFace(img)

    pError = trackFace( info, w, pid, pError)

    #print("Center", info[0], "Area", info[1])

    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break