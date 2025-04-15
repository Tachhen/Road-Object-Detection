from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = os.path.expanduser("~/myProject/GB Road Turns Detection.v5i.yolov8/Output/train/weights/best.pt")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Verify model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Road detection model not found at: {MODEL_PATH}")

# Load models
road_model = YOLO(MODEL_PATH)
vehicle_model = YOLO("yolov8n.pt")  # From static folder

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # Clear previous results
        for f in os.listdir(RESULT_FOLDER):
            os.remove(os.path.join(RESULT_FOLDER, f))
            
        uploaded_files = request.files.getlist('files[]')
        result_images = []
        
        for file in uploaded_files:
            if file.filename == '':
                continue
                
            # Save upload
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            
            # Process image
            img = cv2.imread(upload_path)
            
            # Run detections
            road_results = road_model(img)
            vehicle_results = vehicle_model(img)
            
            # Combine results
            for r in road_results + vehicle_results:
                img = r.plot(img=img)
            
            # Save result
            result_filename = f"processed_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, img)
            result_images.append(result_filename)
        
        return redirect(url_for('show_results', images=','.join(result_images)))
    
    return render_template('upload.html')

@app.route('/results')
def show_results():
    images = request.args.get('images', '').split(',')
    return render_template('results.html', images=[img for img in images if img])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)