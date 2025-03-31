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

# Add CORS support
from flask_cors import CORS
CORS(app, resources={
  r"/*": {
    "origins": "*",  # Or 'https://nice-flower-0dc97321e.6.azurestaticapps.net'
    "allow_headers": ["Content-Type", "Authorization", "bypass-tunnel-reminder"],
    "methods": ["GET", "POST", "PUT", "DELETE"]
  }
})


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
face_detection_enabled = False
tracking_enabled = False
selected_face_id = 1 # Default to first face if theres only one detected
pError = 0  # Previous error for PI controller
fbRange = [5800, 6300]  # Forward-backward range (may need tuning for 320x240)
pid = [0.4, 0.4]  # PI controller gains
w = 320  # Width of the processing frame
current_faces = []  # To store detected faces

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
    global frame, pError, current_faces
    while True:
        new_frame = tello.get_frame_read().frame  # Capture BGR frame
        if new_frame is not None:
            # Resize to 320x240 for processing
            resized_frame_bgr = cv2.resize(new_frame, (320, 240))

            # Convert to RGB for the model
            resized_frame_rgb = cv2.cvtColor(resized_frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize to 640x480 for display
            display_frame = cv2.resize(resized_frame_rgb, (640, 480))

            if face_detection_enabled:
                # Prepare image for YOLO
                pil_img = Image.fromarray(resized_frame_rgb)
                # Run face detection
                output = model(pil_img)
                results = Detections.from_ultralytics(output[0])

                # Process detections to find all faces
                myFaceList = []
                for box in results.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    w_box = x2 - x1
                    h_box = y2 - y1
                    cx = x1 + w_box // 2
                    cy = y1 + h_box // 2
                    area = w_box * h_box
                    myFaceList.append({'box': [x1, y1, x2, y2], 'center': [cx, cy], 'area': area})

                # Sort faces by x-coordinate (left to right)
                myFaceList.sort(key=lambda x: x['center'][0])

                # Assign IDs
                for i, face in enumerate(myFaceList):
                    face['id'] = i + 1

                with lock:
                    current_faces = [{'id': face['id'], 'center': face['center'], 'area': face['area']} for face in myFaceList]

                            # Draw bounding boxes with IDs on display frame
                for face in myFaceList:
                    x1, y1, x2, y2 = face['box']
                    # Scale coordinates to 640x480
                    x1_display = x1 * 2
                    y1_display = y1 * 2
                    x2_display = x2 * 2
                    y2_display = y2 * 2
                    cv2.rectangle(display_frame, (x1_display, y1_display), (x2_display, y2_display), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Face {face['id']}", (x1_display, y1_display - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Control drone if tracking is enabled and a face is selected
                if tracking_enabled and selected_face_id is not None:
                    selected_face = next((face for face in myFaceList if face['id'] == selected_face_id), None)
                    if selected_face:
                        info = [selected_face['center'], selected_face['area']]
                        pError = trackFace(info, w, pid, pError)
                    else:
                        tello.send_rc_control(0, 0, 0, 0)  # Hover if selected face not found
                else:
                    tello.send_rc_control(0, 0, 0, 0)  # Hover when not tracking
            else:
                with lock:
                    current_faces = []

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
@app.route('/v1/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for manual control (disabled during tracking)
@app.route('/v1/control', methods=['POST'])
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
    elif command == 'flip_left':
        tello.flip_left()
    elif command == 'flip_right':
        tello.flip_right()
    elif command == 'flip_forward':
        tello.flip_forward()
    elif command == 'flip_backward':
        tello.flip_back()
    elif command == 'takeoff':
        tello.takeoff()
    elif command == 'land':
        tello.land()
    return 'OK', 200

# Route to toggle face detection
@app.route('/drone/v1/toggle-face-detection', methods=['GET'])
def toggle_face_detection():
    global face_detection_enabled, tracking_enabled
    face_detection_enabled = not face_detection_enabled
    if not face_detection_enabled:
        tracking_enabled = False  # Disable tracking when face detection is off
    print(f"#########################\n          Face detection enabled: {face_detection_enabled}           \n#####################")
    return jsonify({'face_detection': face_detection_enabled, 'tracking': tracking_enabled})

# Route to toggle face tracking
@app.route('/drone/v1/toggle-face-tracking', methods=['GET'])
def toggle_tracking():
    global tracking_enabled
    if not tracking_enabled and not face_detection_enabled:
        return 'Face detection must be enabled to start tracking', 403
    tracking_enabled = not tracking_enabled
    print(f"#########################\n          Tracking enabled: {tracking_enabled}           \n#####################")
    return jsonify({'tracking': tracking_enabled})

# Route to select face for tracking
@app.route('/v1/select_face', methods=['POST'])
def select_face():
    global selected_face_id
    data = request.get_json()
    face_id = data.get('face_id')
    selected_face_id = face_id
    return 'OK', 200

# Route to get telemetry data
@app.route('/v1/telemetry')
def telemetry():
    battery = tello.get_battery()
    height = tello.get_height()
    return jsonify({
        'battery': battery,
        'height': height
    })

# Route to get detected faces
@app.route('/v1/faces')
def faces():
    with lock:
        return jsonify(current_faces)

# @app.after_request
# def add_cors_headers(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
#     return response

# Start the frame capture thread and run the app
if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)

# to run ngrok server:
# ngrok http --url=mallard-learning-feline.ngrok-free.app 5000 

# localtunnel server:
# lt --port 5000 --subdomain comp4537g2