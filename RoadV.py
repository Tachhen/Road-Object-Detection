import os
import shutil
import glob
import subprocess
from ultralytics import YOLO
import torch
import time
import cv2
import numpy as np

DATASET_PATH = r"/home/heathcliff/myProject/GB Road Turns Detection.v5i.yolov8"
DATA_YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")
TEST_MEDIA_DIR = os.path.join(DATASET_PATH, "My Test")  # Changed from TEST_IMAGES_DIR
OUTPUT_DIR = os.path.join(DATASET_PATH, "Output")
PREDICTION_OUTPUT = os.path.join(OUTPUT_DIR, "Predictions")
COMBINED_OUTPUT = os.path.join(OUTPUT_DIR, "Combined_Detections")

VEHICLE_MODEL = "yolov8n.pt"    
TRAFFIC_SIGN_MODEL = "yolov8n.pt"  

def check_gpu_availability():
    """Check GPU status with more detailed info"""
    print("\nHardware Configuration:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}") if torch.cuda.is_available() else print("CUDA: Not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}") if torch.cuda.is_available() else print("GPU: Not detected")
    print(f"RAM: {round(shutil.disk_usage('/').free / (1024**3), 1)}GB free")

def run_yolo_command(command, task_name):
    """Run command with real-time output streaming"""
    print(f"\n Starting {task_name} at {time.strftime('%H:%M:%S')}")
    print(f"Command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
            
        print(f"✅ {task_name} completed at {time.strftime('%H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n {task_name} failed with code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def process_media(turn_model, vehicle_model, media_path, output_dir, is_video=False):
    """Process either an image or video frame"""
    colors = {
        'turn': (0, 255, 0),       # Green for road turns
        'vehicle': (255, 0, 0),     # Blue for vehicles
        'pedestrian': (0, 0, 255),  # Red for pedestrians
        'sign': (255, 255, 0)       # Yellow for signs
    }

    if is_video:
        # Video processing
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"Error opening video file: {media_path}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Prepare video writer
        output_path = os.path.join(output_dir, os.path.basename(media_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = process_frame(turn_model, vehicle_model, frame, colors)
            out.write(processed_frame)
            frame_count += 1

            # Display progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"Saved processed video: {output_path} ({frame_count} frames)")
    else:
        # Image processing
        img = cv2.imread(media_path)
        if img is None:
            print(f"Error loading image: {media_path}")
            return

        processed_img = process_frame(turn_model, vehicle_model, img, colors)
        output_path = os.path.join(output_dir, os.path.basename(media_path))
        cv2.imwrite(output_path, processed_img)
        print(f"Saved processed image: {output_path}")

def process_frame(turn_model, vehicle_model, frame, colors):
    """Process a single frame with both models"""
    # Process road turns
    turn_results = turn_model(frame)
    for result in turn_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors['turn'], 2)
            cv2.putText(frame, f"Turn: {result.names[int(box.cls)]}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['turn'], 2)
    
    # Process vehicles and pedestrians
    vehicle_results = vehicle_model(frame)
    for result in vehicle_results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = result.names[class_id]
            
            if label in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors['pedestrian'] if label == 'person' else colors['vehicle']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def combine_detections(turn_model_path, test_media_dir, output_dir):
    """Combine road turn, vehicle, pedestrian and traffic sign detections"""
    print("\n🔗 Combining all detections...")
    
    # Load models
    turn_model = YOLO(turn_model_path)
    vehicle_model = YOLO(VEHICLE_MODEL)  
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported media extensions
    media_extensions = ['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi', '*.mov']
    media_paths = []
    for ext in media_extensions:
        media_paths.extend(glob.glob(os.path.join(test_media_dir, ext)))
    
    for media_path in media_paths:
        is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov'))
        process_media(turn_model, vehicle_model, media_path, output_dir, is_video)
    
    print(f"\nCombined {len(media_paths)} media files with all detections")

def main():
    print("\nVerifying paths...")
    for name, path in [
        ("Dataset", DATASET_PATH),
        ("Test media", TEST_MEDIA_DIR),
        ("YAML config", DATA_YAML_PATH)
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} path not found: {path}")
        print(f"✓ {name}: {path}")

    check_gpu_availability()

    os.makedirs(PREDICTION_OUTPUT, exist_ok=True)
    os.makedirs(COMBINED_OUTPUT, exist_ok=True)

    train_cmd = [
        "yolo", "task=detect", "mode=train",
        "model=yolov8n.pt",
        f"data={DATA_YAML_PATH}",
        "epochs=50", "imgsz=800",
        "device=0",  
        "batch=8",   
        f"project={OUTPUT_DIR}",
        "name=train",
        "exist_ok=True"
    ]

    if not run_yolo_command(train_cmd, "Training"):
        print("\nTraining failed. Cannot proceed with predictions.")
        return

    predict_cmd = [
        "yolo", "task=detect", "mode=predict",
        f"model={os.path.join(OUTPUT_DIR, 'train', 'weights', 'best.pt')}",
        f"source={TEST_MEDIA_DIR}",
        "save=True", "device=0",
        f"project={PREDICTION_OUTPUT}",
        "name=results"
    ]

    if not run_yolo_command(predict_cmd, "Prediction"):
        print("\nPrediction failed.")
        return

    turn_model_path = os.path.join(OUTPUT_DIR, "train", "weights", "best.pt")
    combine_detections(turn_model_path, TEST_MEDIA_DIR, COMBINED_OUTPUT)

    results = glob.glob(os.path.join(COMBINED_OUTPUT, "*"))
    print(f"\nFinal results with combined detections:")
    for result in results[:3]:  # Show first 3 results
        print(f"- {os.path.basename(result)}")

if __name__ == "__main__":
    print("Starting Enhanced Road Detection Pipeline")
    try:
        main()
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
    print("\nScript completed")