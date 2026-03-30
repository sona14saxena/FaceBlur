# Privacy Face Blur

A real-time face detection and obscuring script built with Python, OpenCV, and Mediapipe.

This tool automatically detects faces from a live webcam feed and applies a heavy Gaussian blur to the face regions to protect privacy. It acts entirely on edge (locally on your machine) and uses Google's modern Mediapipe Tasks API.

## Requirements
- Python 3.12 (or any compatible recent Python version)
- A connected webcam

## Installation

Install the required dependencies:
```bash
pip install opencv-python mediapipe
```

## Usage

Simply run the main script:
```bash
python main.py
```
*Note: On its first run, the script will automatically download the official light-weight Mediapipe face detection model (`blaze_face_short_range.tflite`) into the local directory.*

Press the **ESC** key while selecting the webcam window to exit the application properly.
