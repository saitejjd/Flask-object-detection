import os
import cv2
import threading
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'heic'}

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Create directories if missing
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Thread-safe camera control
camera_lock = threading.Lock()
camera_active = False
cap = None

# COCO class names (80 classes)
COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
    "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames():
    """Generate frames from the webcam for live detection."""
    global camera_active, cap
    with camera_lock:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Camera unavailable")
            
            camera_active = True
            while camera_active:
                success, frame = cap.read()
                if not success:
                    break

                # Run object detection
                results = model.predict(frame, verbose=False)
                annotated_frame = results[0].plot()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        finally:
            if cap and cap.isOpened():
                cap.release()
            camera_active = False

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    """Start the camera feed."""
    global camera_active
    with camera_lock:
        if not camera_active:
            camera_active = True
    return "Camera started"

@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    if not camera_active:
        return "Camera not active", 403
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera feed."""
    global camera_active
    with camera_lock:
        camera_active = False
    return "Camera stopped"

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and run object detection."""
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    if not allowed_file(file.filename):
        return "Only JPG/PNG/HEIC images allowed", 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert HEIC to JPG if needed
        if filename.lower().endswith('.heic'):
            from pillow_heif import register_heif_opener
            from PIL import Image
            register_heif_opener()
            img = Image.open(filepath)
            filename = filename.rsplit('.', 1)[0] + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath, "JPEG")

        # Run object detection
        results = model.predict(
            filepath,
            save=True,
            project='static',
            name='results',
            exist_ok=True  # Prevent numbered folders
        )

        # Get detection results
        detections = [
            f"{COCO_NAMES[int(box.cls[0])]}: {float(box.conf[0]):.2f}"
            for box in results[0].boxes
        ]

        # Get output image path
        output_filename = os.path.basename(results[0].save_dir) + '.jpg'
        output_path = os.path.join('results', output_filename)

        return render_template('index.html',
                            input_image=f"uploads/{filename}",
                            output_image=output_path,
                            detections=detections)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)