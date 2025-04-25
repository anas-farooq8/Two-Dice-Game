# Two Dice Game

A computer vision-based dice recognition game using YOLOv8, Flask, and IP Webcam.

## Overview

The Two Dice Game is an interactive web application that uses computer vision to detect and recognize dice values from a camera feed. Players place one die on each side of the camera view, and the application detects the values, calculates scores, and determines when the game ends.

## Game Rules

1. Two players (left and right) take turns rolling dice.
2. Each player places one die on their respective side of the camera view.
3. The application recognizes the dice values and adds them to the player's score.
4. If both dice show the same value, the game ends.
5. The player with the highest score wins.

## Features

- Real-time dice detection using a pre-trained YOLOv8 model
- Live camera feed using IP Webcam
- Automatic scoring system
- Multi-attempt detection system for improved accuracy
- Game state management
- Responsive web interface
- Mobile-friendly design
- Error handling and connection monitoring

## Technical Stack

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Frontend**: HTML, CSS, JavaScript
- **Camera**: IP Webcam app (Android)

## Usage

### Setting Up the Camera

1. Install the IP Webcam app on your Android device
2. Open the app and start the server
3. Note the IP address displayed in the app (e.g., http://192.168.1.73:8080)

### Running the Application

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Open a web browser and navigate to:

   ```
   http://localhost:5000
   ```

3. Enter your IP Webcam URL in the format:

   ```
   http://IP_ADDRESS:8080/shot.jpg
   ```

   For example: `http://192.168.1.73:8080/shot.jpg`

4. Click "Update Camera" to connect to your IP Webcam

5. Click "Start Game" to begin playing

6. Position one die on each side of the camera view and click "Capture"

### Game Interface

- **Camera Feed**: Shows the live feed from your IP Webcam with left and right sides marked
- **Score Board**: Displays current scores for both players
- **Game Status**: Shows game messages and instructions
- **Dice Display**: Shows the most recently captured dice values
- **Last Capture**: Displays the processed image with dice detection boxes

## Model Training

The dice detection model was trained using YOLOv8 on a dataset of dice images. The training process involves:

1. Data preparation: Organizing images and annotations
2. Model selection: Using YOLOv8x (extra large) pre-trained model
3. Training configuration: Setting epochs, batch size, and image size
4. Training: Running the model training process
5. Evaluation: Testing model performance on test data
6. Export: Saving the trained model for use in the application

### Training Details

- **Model Architecture**: YOLOv8x
- **Training Data**: Annotated dice images
- **Classes**: 6 (dice faces 1-6)
- **Epochs**: 20
- **Image Size**: 640x640
- **Batch Size**: 16

The training code can be found in `train.py`.
![Image](https://github.com/user-attachments/assets/5c1b9a4f-346a-44b9-96b3-809ba42810b4)

## Implementation Details

### Dice Detection Process

1. **Image Acquisition**: Frames are captured from the IP Webcam
2. **Image Enhancement**: Contrast adjustment and sharpening for better recognition
3. **Multi-attempt Detection**:
   - Standard prediction
   - Brightness-adjusted prediction
   - Perspective-transformed prediction
4. **Duplicate Filtering**: Removing duplicate detections of the same dice
5. **Confidence Thresholding**: Using confidence scores to validate detections
6. **Historical Value Tracking**: Remembering previous valid detections for reliability

### Key Algorithms

#### Dice Filtering

```python
def filter_duplicate_dice(dice_list):
    # Sort by confidence (descending)
    dice_list.sort(key=lambda x: x[1], reverse=True)

    # Filter unique dice values
    unique_dice = {}
    for value, confidence, box in dice_list:
        if value not in unique_dice:
            unique_dice[value] = (value, confidence, box)

    # Return the list of unique dice
    return list(unique_dice.values())
```

#### Image Enhancement

```python
# Apply image enhancement
alpha = 1.2  # Contrast control
beta = 5     # Brightness control
frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Apply slight sharpening
kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
frame = cv2.filter2D(frame, -1, kernel)
```

## Troubleshooting

### Camera Connection Issues

- Ensure your phone and computer are on the same network
- Check that your firewall isn't blocking the connection
- Verify the IP address is correct in the IP Webcam app
- Try disabling mobile data on your phone

### Dice Detection Problems

- Ensure good lighting conditions
- Use dice with high contrast between dots and background
- Place dice on a plain, non-reflective surface
- Keep dice separated with clear space between them
- Try adjusting the camera angle if detection fails

### Performance Issues

- For better performance, use a computer with a CUDA-compatible GPU
- Close other resource-intensive applications
- Reduce the camera resolution in the IP Webcam app if the stream is laggy

### Demo

https://github.com/user-attachments/assets/77fbfe79-26da-4cc5-9353-ab7d09351728
