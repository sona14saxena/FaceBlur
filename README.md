# FACEBLUR: Real-Time Face Redaction for Community Safety

FACEBLUR is an AI-powered "Privacy-by-Design" tool developed to automate facial anonymization in live video feeds. Built using Python, OpenCV, and Mediapipe, this project addresses the ethical challenges of documenting community events while protecting the identities of vulnerable individuals.

## The Problem & Community Impact
In environments like NGO field visits, capturing media often inadvertently exposes the faces of bystanders.
- **Vulnerable Groups**: For women in sensitive situations or children in public care, revealing an identity online can lead to safety risks.
- **The Solution**: FACEBLUR ensures that "Consent is the Default." By blurring faces in real-time, the tool allows for the documentation of "Actions" and "Impact" without compromising the "Individual."

## Technical Features
- **High-Speed Detection**: Uses the BlazeFace Short-Range model for sub-millisecond inference.
- **Optimized for Edge**: Designed to run at 30+ FPS on standard student laptops without requiring a GPU.
- **Privacy-First**: The Gaussian Blur ($99 \times 99$ kernel) is applied locally; un-blurred data is never stored on the disk.
- **Secure Capture**: Includes a one-touch feature to save redacted screenshots for social media.

## Installation & Setup

1. **Clone the Repository**:
```bash
git clone https://github.com/sona14saxena/FaceBlur.git
cd FaceBlur
```

2. **Install Dependencies**:
```bash
pip install opencv-python mediapipe
```

3. **Run the Application**:
```bash
python src/main.py
```
*Note: On the first run, the script automatically downloads the `blaze_face_short_range.tflite` model.*

## How to Use
- **Live Feed**: The webcam window will open with automatic blurring enabled.
- **Save Redacted Frame**: Press `S` to save a privacy-protected image to the `output/` folder.
- **Exit**: Press `Q` or `ESC` to safely close the application.

## Project Structure
- `src/main.py`: The core detection and processing logic.
- `output/`: Directory where privacy-protected screenshots are stored.
- `docs/`: Contains the full Project Report and technical documentation.

## Evaluation & Academic Context
This project was developed as a Bring Your Own Project (BYOP) capstone. It demonstrates:
- **Relevance**: Application of Digital Image Processing and Object Detection.
- **Originality**: Purposeful use-case for Women and Children's safety.
- **Quality**: Organized, modular code following SDC (Software Development Club) standards.
