import os
import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import random

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configuration
MODEL_PATH = 'model/dice_detector.pt'
IPWEBCAM_URL = 'http://192.168.1.73:8080/shot.jpg'
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for dice detection
IMAGE_QUALITY = 100  # Highest quality for JPEG encoding (0-100)
MAX_IMAGE_SIZE = 1600  # Increased max dimension for better resolution
IOU_THRESHOLD = 0.45  # IOU threshold for non-maximum suppression
MAX_RETRY_ATTEMPTS = 3  # Number of attempts to detect dice

# Game state
game_state = {
    'active': False,
    'left_score': 0,
    'right_score': 0,
    'game_over': False,
    'message': 'Click "Start Game" to begin',
    'last_capture': None,
    'left_dice': [],
    'right_dice': [],
    'round': 0,
    'last_left_dice': None,  # Store last detected left dice value
    'last_right_dice': None  # Store last detected right dice value
}

# Load the model
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_camera_frame():
    """Get a frame from IP Webcam"""
    try:
        # Add a random timestamp to prevent caching
        url_with_timestamp = f"{IPWEBCAM_URL}?t={int(time.time())}&rand={random.randint(1, 10000)}"
        # Get the image from the IP Webcam
        response = requests.get(url_with_timestamp, timeout=8)
        if response.status_code == 200:
            # Convert the image to OpenCV format
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Check if image was loaded correctly
            if frame is None or frame.size == 0:
                print("Received empty frame from camera")
                return None
                
            # Resize if too large (preserving aspect ratio)
            h, w = frame.shape[:2]
            if max(h, w) > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Apply image enhancement
            # Adjust contrast and brightness
            alpha = 1.2  # Contrast control (1.0 means no change)
            beta = 5     # Brightness control (0 means no change)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], 
                               [-1, 9,-1],
                               [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)
                
            return frame
        else:
            print(f"Failed to get frame, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting camera frame: {e}")
        return None

def process_frame(frame):
    """Process the camera frame to detect dice and calculate scores"""
    if frame is None or model is None:
        return None, "Failed to process frame or model not loaded"
    
    # Create a copy of the frame for annotation
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    mid_point = width // 2
    
    # Draw a line to divide left and right sides
    cv2.line(annotated_frame, (mid_point, 0), (mid_point, height), (0, 255, 0), 2)
    
    # Lists to store detected dice information
    left_dice = []
    right_dice = []
    
    # Run multiple model predictions with different parameters to improve detection
    success = False
    
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            # Try with different confidence thresholds
            conf_threshold = CONFIDENCE_THRESHOLD - (0.05 * attempt)
            # Try different augmentations for each attempt
            if attempt == 0:
                # Standard prediction
                results = model.predict(frame, conf=conf_threshold, iou=IOU_THRESHOLD)
            elif attempt == 1:
                # Try with slightly brighter image
                brightened = cv2.convertScaleAbs(frame, alpha=1.3, beta=10)
                results = model.predict(brightened, conf=conf_threshold, iou=IOU_THRESHOLD)
            else:
                # Try with slightly different perspective
                h, w = frame.shape[:2]
                src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                dst_points = np.float32([[0, 0], [w, 0], [w, h+20], [0, h-20]])
                perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                warped = cv2.warpPerspective(frame, perspective_matrix, (w, h))
                results = model.predict(warped, conf=conf_threshold, iou=IOU_THRESHOLD)
            
            # Process results
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Only consider results if we found any boxes
                if len(boxes) > 0:
                    for box, cls, conf in zip(boxes, classes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        dice_value = int(cls) + 1  # Convert from 0-indexed to 1-6
                        
                        # Determine if dice is on left or right side
                        if center_x < mid_point:
                            left_dice.append((dice_value, conf, box))
                        else:
                            right_dice.append((dice_value, conf, box))
                    
                    # If we have at least one dice on each side, consider it a success
                    if len(left_dice) > 0 and len(right_dice) > 0:
                        success = True
                        break
                
        except Exception as e:
            print(f"Error during prediction attempt {attempt+1}: {e}")
    
    # Draw any existing detections on the frame
    # Process left side detections
    left_dice_filtered = filter_duplicate_dice(left_dice)
    for value, conf, box in left_dice_filtered:
        x1, y1, x2, y2 = map(int, box)
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add text label
        label = f"{value} ({conf:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Process right side detections
    right_dice_filtered = filter_duplicate_dice(right_dice)
    for value, conf, box in right_dice_filtered:
        x1, y1, x2, y2 = map(int, box)
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add text label
        label = f"{value} ({conf:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    message = ""
    
    # Debug information
    cv2.putText(annotated_frame, f"Left dice: {len(left_dice_filtered)}, Right dice: {len(right_dice_filtered)}", 
               (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Check detection status
    if len(left_dice_filtered) != 1 or len(right_dice_filtered) != 1:
        # Handle case when we don't have exactly one dice on each side
        if len(left_dice_filtered) == 0 and game_state['last_left_dice'] is not None:
            # Use the last detected left dice value
            left_dice_filtered = [(game_state['last_left_dice'], 1.0, [0, 0, 0, 0])]
            message += "Using previous left dice value. "
        
        if len(right_dice_filtered) == 0 and game_state['last_right_dice'] is not None:
            # Use the last detected right dice value
            right_dice_filtered = [(game_state['last_right_dice'], 1.0, [0, 0, 0, 0])]
            message += "Using previous right dice value. "
        
        # If we still don't have one dice on each side
        if len(left_dice_filtered) != 1 or len(right_dice_filtered) != 1:
            message += f"Detected {len(left_dice_filtered)} dice on left and {len(right_dice_filtered)} on right. Please position exactly one dice on each side."
            return annotated_frame, message
    
    # Get the dice values and check if they're the same
    left_value = left_dice_filtered[0][0]
    right_value = right_dice_filtered[0][0]
    
    # Store the detected dice values for future reference
    game_state['last_left_dice'] = left_value
    game_state['last_right_dice'] = right_value
    
    if left_value == right_value:
        game_state['game_over'] = True
        message = f"Game Over! Both dice show {left_value}. Final scores: Left: {game_state['left_score']}, Right: {game_state['right_score']}"
    else:
        # Add scores
        game_state['left_score'] += left_value
        game_state['right_score'] += right_value
        game_state['round'] += 1
        message = f"Left dice: {left_value}, Right dice: {right_value}. Updated scores: Left: {game_state['left_score']}, Right: {game_state['right_score']}"
    
    # Add round number to frame
    cv2.putText(annotated_frame, f"Round: {game_state['round']}", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add current scores to frame
    cv2.putText(annotated_frame, f"Left score: {game_state['left_score']}", 
               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Right score: {game_state['right_score']}", 
               (mid_point + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Update game state
    game_state['message'] = message
    game_state['left_dice'] = [i[0] for i in left_dice_filtered]  # Store only the values
    game_state['right_dice'] = [i[0] for i in right_dice_filtered]  # Store only the values
    
    return annotated_frame, message

def filter_duplicate_dice(dice_list):
    """Filter duplicates, keeping only the detection with highest confidence for each dice value"""
    if not dice_list:
        return []
        
    # Sort by confidence (descending)
    dice_list.sort(key=lambda x: x[1], reverse=True)
    
    # If there's only one detection, return it
    if len(dice_list) == 1:
        return dice_list
        
    # Filter unique dice values
    unique_dice = {}
    for value, confidence, box in dice_list:
        if value not in unique_dice:
            unique_dice[value] = (value, confidence, box)
    
    # Return the list of unique dice
    return list(unique_dice.values())

@app.route('/')
def index():
    """Render the main game page"""
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    """Start or reset the game"""
    game_state['active'] = True
    game_state['left_score'] = 0
    game_state['right_score'] = 0
    game_state['game_over'] = False
    game_state['message'] = 'Game started! Position one dice on each side and click "Capture"'
    game_state['last_capture'] = None
    game_state['left_dice'] = []
    game_state['right_dice'] = []
    game_state['round'] = 0
    game_state['last_left_dice'] = None
    game_state['last_right_dice'] = None
    
    return jsonify({'status': 'success', 'message': game_state['message']})

@app.route('/capture', methods=['POST'])
def capture():
    """Capture and process a frame from the camera"""
    if not game_state['active']:
        return jsonify({'status': 'error', 'message': 'Game not active. Click "Start Game" first.'})
    
    if game_state['game_over']:
        return jsonify({'status': 'error', 'message': 'Game is over. Click "Start Game" to play again.'})
    
    # Get frame from camera
    frame = get_camera_frame()
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Failed to capture image. Check camera connection.'})
    
    # Process the frame
    annotated_frame, message = process_frame(frame)
    
    # Convert the annotated frame to base64 for sending to the browser
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY]
    _, img_encoded = cv2.imencode('.jpg', annotated_frame, encode_params)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    game_state['last_capture'] = img_base64
    
    # Return the processed result
    return jsonify({
        'status': 'success',
        'message': message,
        'image': img_base64,
        'left_score': game_state['left_score'],
        'right_score': game_state['right_score'],
        'game_over': game_state['game_over'],
        'round': game_state['round'],
        'left_dice': game_state['left_dice'],
        'right_dice': game_state['right_dice']
    })

@app.route('/video_feed')
def video_feed():
    """Streaming route for the live camera feed"""
    def generate():
        while True:
            try:
                frame = get_camera_frame()
                if frame is not None:
                    # Draw line to show left/right sides
                    height, width = frame.shape[:2]
                    mid_point = width // 2
                    cv2.line(frame, (mid_point, 0), (mid_point, height), (0, 255, 0), 2)
                    
                    # Add labels for left and right players
                    cv2.putText(frame, 'Left Player', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, 'Right Player', (mid_point + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Encode the frame with higher quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_params)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Brief delay to control frame rate and reduce CPU usage
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in video feed: {e}")
                # Return an error image
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, "Camera Error - Check Connection", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Wait longer before retrying
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_camera_url', methods=['POST'])
def update_camera_url():
    """Update the IP Webcam URL"""
    global IPWEBCAM_URL
    data = request.json
    new_url = data.get('url')
    
    if new_url:
        IPWEBCAM_URL = new_url
        # Try to get a test frame to validate the URL
        try:
            frame = get_camera_frame()
            if frame is not None:
                return jsonify({'status': 'success', 'message': f'Camera URL updated to {new_url}'})
            else:
                return jsonify({'status': 'error', 'message': 'Connected to URL but received no valid image'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error connecting to camera: {str(e)}'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid URL provided'})

@app.route('/game_status')
def game_status():
    """Return the current game status"""
    return jsonify({
        'active': game_state['active'],
        'left_score': game_state['left_score'],
        'right_score': game_state['right_score'],
        'game_over': game_state['game_over'],
        'message': game_state['message'],
        'last_capture': game_state['last_capture'],
        'round': game_state['round'],
        'left_dice': game_state['left_dice'],
        'right_dice': game_state['right_dice']
    })

if __name__ == '__main__':
    print("Starting web server...")
    app.run(host='0.0.0.0', port=5000, debug=True) 