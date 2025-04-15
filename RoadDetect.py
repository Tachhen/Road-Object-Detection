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
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "My Test")
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

def combine_detections(turn_model_path, test_images_dir, output_dir):
    """Combine road turn, vehicle, pedestrian and traffic sign detections"""
    print("\n🔗 Combining all detections...")
    
 
    turn_model = YOLO(turn_model_path)
    vehicle_model = YOLO(VEHICLE_MODEL)  
    
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(test_images_dir, "*.[jp][pn]g"))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        
        turn_results = turn_model(img)
        vehicle_results = vehicle_model(img)
        
        
        
        colors = {
            'turn': (0, 255, 0),       # Green for road turns
            'vehicle': (255, 0, 0),     # Blue for vehicles
            'pedestrian': (0, 0, 255), # Red for pedestrians
            'sign': (255, 255, 0)       # Yellow for signs
        }
        

        for result in turn_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), colors['turn'], 2)
                cv2.putText(img, f"Turn: {result.names[int(box.cls)]}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['turn'], 2)
        
       
        for result in vehicle_results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = result.names[class_id]
                
             
                if label in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = colors['pedestrian'] if label == 'person' else colors['vehicle']
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        print(f"Saved combined detection: {output_path}")
    
    print(f"\nCombined {len(image_paths)} images with all detections")

def main():
    
    print("\n Verifying paths...")
    for name, path in [
        ("Dataset", DATASET_PATH),
        ("Test images", TEST_IMAGES_DIR),
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
        f"source={TEST_IMAGES_DIR}",
        "save=True", "device=0",
        f"project={PREDICTION_OUTPUT}",
        "name=results"
    ]

    if not run_yolo_command(predict_cmd, "Prediction"):
        print("\nPrediction failed.")
        return


    turn_model_path = os.path.join(OUTPUT_DIR, "train", "weights", "best.pt")
    combine_detections(turn_model_path, TEST_IMAGES_DIR, COMBINED_OUTPUT)


    results = glob.glob(os.path.join(COMBINED_OUTPUT, "*.jpg"))
    print(f"\nFinal results with combined detections:")
    for img in results[:3]:  
        print(f"- {os.path.basename(img)}")

if __name__ == "__main__":
    print(" Starting Enhanced Road Detection Pipeline")
    try:
        main()
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
    print("\nScript completed")