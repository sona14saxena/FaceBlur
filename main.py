import cv2, urllib.request, os, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Download the required model if it's missing (needed for Mediapipe on newer Python versions)
if not os.path.exists('face.tflite'):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite', 'face.tflite')

# Initialize the Face Detector
opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path='face.tflite'))
with vision.FaceDetector.create_from_options(opts) as fd:
    cap = cv2.VideoCapture(0)
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
            
        cv2.imshow('Face Blur', frame)
        if cv2.waitKey(1) == 27: break # Press ESC to exit
    cap.release()
cv2.destroyAllWindows()
