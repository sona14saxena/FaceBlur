import cv2, urllib.request, os, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from datetime import datetime

# Download the required model if it's missing (needed for Mediapipe on newer Python versions)
if not os.path.exists('face.tflite'):
    print("Downloading Face Detection Model...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite', 'face.tflite')

# Ensure output directory exists for saved screenshots
os.makedirs('output', exist_ok=True)

# Initialize the Face Detector
opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path='face.tflite'))
with vision.FaceDetector.create_from_options(opts) as fd:
    cap = cv2.VideoCapture(0)
    print("PrivacyLens Started.\nPress 'S' to save a redacted frame.\nPress 'Q' or 'ESC' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Detect faces
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for det in fd.detect(mp_img).detections:
            b = det.bounding_box
            x, y, bw, bh = max(0, b.origin_x), max(0, b.origin_y), b.width, b.height
            # Apply heavy Gaussian blur to the detected face region
            if bw > 0 and bh > 0: frame[y:y+bh, x:x+bw] = cv2.GaussianBlur(frame[y:y+bh, x:x+bw], (99, 99), 30)
            
        cv2.imshow('PrivacyLens', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'): 
            break # ESC or Q to exit
        elif key == ord('s') or key == ord('S'):
            # Save frame to output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join('output', f'redacted_{timestamp}.png')
            cv2.imwrite(filename, frame)
            print(f"[{timestamp}] Frame saved securely to: {filename}")
            
    cap.release()
cv2.destroyAllWindows()
