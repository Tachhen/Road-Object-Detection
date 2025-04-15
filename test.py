import os
import cv2
import glob
import torch
from ultralytics import YOLO
import numpy as np

# Configuration
DATASET_PATH = "/home/heathcliff/myProject/GB Road Turns Detection.v5i.yolov8"
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "My Test")
OUTPUT_DIR = os.path.join(DATASET_PATH, "Output")
PREDICTION_OUTPUT = os.path.join(OUTPUT_DIR, "Combined_Results")
os.makedirs(PREDICTION_OUTPUT, exist_ok=True)

# COCO classes to detect (person, car, truck, etc.)
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def setup_models():
    """Initialize both models with GPU if available"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load your custom road turn model
    road_model = YOLO(os.path.join(OUTPUT_DIR, "train", "weights", "best.pt")).to(device)
    
    # Load pretrained COCO model
    coco_model = YOLO("yolov8n.pt").to(device)
    
    return road_model, coco_model

def visualize_detections(image, road_results, coco_results):
    """Draw both road turns and COCO detections on image"""
    # Road turns (green boxes)
    for box in road_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"Turn {cls} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # COCO objects (red boxes)
    for box in coco_results[0].boxes:
        cls = int(box.cls[0])
        if cls not in COCO_CLASSES:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"{COCO_CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image

def process_images(road_model, coco_model):
    """Process all test images with both models"""
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*"))
    
    for img_path in test_images:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Run both models
        road_results = road_model(img_path)
        coco_results = coco_model(img_path, classes=list(COCO_CLASSES.keys()))
        
        # Combine and save
        img = cv2.imread(img_path)
        result_img = visualize_detections(img, road_results, coco_results)
        
        output_path = os.path.join(PREDICTION_OUTPUT, os.path.basename(img_path))
        cv2.imwrite(output_path, result_img)
        print(f"Saved combined result to {output_path}")

def main():
    # Verify paths
    if not os.path.exists(TEST_IMAGES_DIR):
        raise FileNotFoundError(f"Test images directory not found: {TEST_IMAGES_DIR}")
    
    # Hardware check
    print(f"\n{'='*40}")
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"{'='*40}\n")
    
    # Load models
    road_model, coco_model = setup_models()
    
    # Process images
    process_images(road_model, coco_model)
    
    print("\n✅ All images processed successfully!")

if __name__ == "__main__":
    main()