from flask import Flask, Response, request, render_template, jsonify
from djitellopy import Tello
import cv2
import threading
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np

# Initialize Flask app and Tello drone
app = Flask(__name__)
# from flask_cors import CORS
# CORS(app)
# Add CORS support

tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery: {tello.get_battery()}%")

# Load the YOLO face detection model
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path)

# Global variables
frame = None
lock = threading.Lock()
tracking_enabled = False
pError = 0  # Previous error for PI controller
fbRange = [6200, 6800]  # Forward-backward range (may need tuning for 320x240)
pid = [0.4, 0.4]  # PI controller gains
w = 320  # Width of the processing frame

# Function to control drone based on face position
def trackFace(info, w, pid, pError):
    area = info[1]  # Face area
    x, y = info[0]  # Face center coordinates
    fb = 0  # Forward-backward speed
    error = x - w // 2  # Error from center
    speed = pid[0] * error + pid[1] * (error - pError)  # PI controller
    speed = int(np.clip(speed, -100, 100))  # Yaw speed

    # Adjust forward-backward based on face size
    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20  # Move backward
    elif area < fbRange[0] and area != 0:
        fb = 20  # Move forward

    # If no face detected, stop movement
    if x == 0:
        speed = 0
        error = 0

    tello.send_rc_control(0, fb, 0, speed)  # Send control: left-right, forward-back, up-down, yaw
    return error

# Background thread to capture and process frames
def capture_frames():
    global frame, pError
    while True:
        new_frame = tello.get_frame_read().frame  # Capture BGR frame
        if new_frame is not None:
            # Resize to 320x240 for processing
            resized_frame_bgr = cv2.resize(new_frame, (320, 240))
            # Convert to RGB for the model
            resized_frame_rgb = cv2.cvtColor(resized_frame_bgr, cv2.COLOR_BGR2RGB)
            # Prepare image for YOLO
            pil_img = Image.fromarray(resized_frame_rgb)
            # Run face detection
            output = model(pil_img)
            results = Detections.from_ultralytics(output[0])

            # Process detections to find the largest face
            myFaceListC = []
            myFaceListArea = []
            for box in results.xyxy:
                x1, y1, x2, y2 = map(int, box)
                w_box = x2 - x1
                h_box = y2 - y1
                cx = x1 + w_box // 2
                cy = y1 + h_box // 2
                area = w_box * h_box
                myFaceListC.append([cx, cy])
                myFaceListArea.append(area)

            if len(myFaceListArea) != 0:
                i = myFaceListArea.index(max(myFaceListArea))
                info = [myFaceListC[i], myFaceListArea[i]]
            else:
                info = [[0, 0], 0]

            # Control drone if tracking is enabled
            if tracking_enabled:
                pError = trackFace(info, w, pid, pError)
            else:
                tello.send_rc_control(0, 0, 0, 0)  # Hover when not tracking

            # Resize to 640x480 for display
            display_frame = cv2.resize(resized_frame_rgb, (640, 480))
            # Draw bounding boxes on display frame
            for box in results.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # Scale coordinates to 640x480
                x1 *= 2
                y1 *= 2
                x2 *= 2
                y2 *= 2
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Green center

            with lock:
                frame = display_frame
        time.sleep(0.05)  # Limit to ~20 FPS

# Generator function for streaming video
def generate_frames():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03)  # Control streaming frame rate

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for manual control (disabled during tracking)
@app.route('/control', methods=['POST'])
def control():
    if tracking_enabled:
        return 'Manual controls disabled while tracking', 403
    data = request.get_json()
    command = data.get('command')
    speed = 50
    if command == 'left':
        tello.send_rc_control(-speed, 0, 0, 0)
    elif command == 'right':
        tello.send_rc_control(speed, 0, 0, 0)
    elif command == 'forward':
        tello.send_rc_control(0, speed, 0, 0)
    elif command == 'backward':
        tello.send_rc_control(0, -speed, 0, 0)
    elif command == 'up':
        tello.send_rc_control(0, 0, speed, 0)
    elif command == 'down':
        tello.send_rc_control(0, 0, -speed, 0)
    elif command == 'rotate_left':
        tello.send_rc_control(0, 0, 0, -speed)
    elif command == 'rotate_right':
        tello.send_rc_control(0, 0, 0, speed)
    elif command == 'takeoff':
        tello.takeoff()
    elif command == 'land':
        tello.land()
    return 'OK', 200

# Route to toggle face tracking
@app.route('/toggle_tracking', methods=['GET'])
def toggle_tracking():
    global tracking_enabled
    tracking_enabled = not tracking_enabled
    return jsonify({'tracking': tracking_enabled})

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Start the frame capture thread and run the app
if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)