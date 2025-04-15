import os
import cv2
import glob
import torch
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Configuration
DATASET_PATH = "/home/heathcliff/myProject/GB Road Turns Detection.v5i.yolov8"
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "My Test")
OUTPUT_DIR = os.path.join(DATASET_PATH, "Output")
PREDICTION_OUTPUT = os.path.join(OUTPUT_DIR, "Enhanced_Results")
os.makedirs(PREDICTION_OUTPUT, exist_ok=True)

# Detection configuration
ROAD_CLASSES = {
    0: 'Left-turn',
    1: 'right_turn',
    2: 'straight_road'
}

COCO_CLASSES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

COLORS = {
    'road': (0, 255, 0),       # Green for road signs
    'vehicle': (0, 0, 255),    # Red for vehicles
    'person': (255, 0, 0)      # Blue for persons
}

def setup_models():
    """Initialize models with optimized settings"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models with half-precision for faster inference
    road_model = YOLO(os.path.join(OUTPUT_DIR, "train", "weights", "best.pt")).to(device)
    coco_model = YOLO("yolov8n.pt").to(device)
    
    # Warm up models
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    _ = road_model(dummy_input)
    _ = coco_model(dummy_input)
    
    return road_model, coco_model

def process_detections(results, model_type):
    """Process and filter detections"""
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter by model type
            if model_type == 'road' and cls in ROAD_CLASSES:
                label = f"{ROAD_CLASSES[cls]} {conf:.2f}"
                color = COLORS['road']
                detections.append((box.xyxy[0], label, color))
            elif model_type == 'coco':
                if cls == 0:  # Person
                    label = f"person {conf:.2f}"
                    color = COLORS['person']
                    detections.append((box.xyxy[0], label, color))
                elif cls in COCO_CLASSES:  # Vehicles
                    label = f"{COCO_CLASSES[cls]} {conf:.2f}"
                    color = COLORS['vehicle']
                    detections.append((box.xyxy[0], label, color))
    
    return detections

def visualize_detections(image, road_detections, coco_detections):
    """Draw all detections with improved visualization"""
    # Create copy of image
    vis_img = image.copy()
    
    # Draw all detections
    for detections in [road_detections, coco_detections]:
        for (x1, y1, x2, y2), label, color in detections:
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_img, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add legend
    legend_y = 30
    for label, color in [('Road Signs', COLORS['road']),
                        ('Vehicles', COLORS['vehicle']),
                        ('Persons', COLORS['person'])]:
        cv2.putText(vis_img, label, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        legend_y += 30
    
    return vis_img

def process_images(road_model, coco_model):
    """Process all test images with enhanced detection"""
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*"))
    
    for img_path in test_images:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Read image once
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue
        
        # Run models with optimal image size
        road_results = road_model(img, imgsz=640, conf=0.5)
        coco_results = coco_model(img, imgsz=640, conf=0.5, classes=list(COCO_CLASSES.keys()))
        
        # Process detections
        road_detections = process_detections(road_results, 'road')
        coco_detections = process_detections(coco_results, 'coco')
        
        # Generate visualization
        result_img = visualize_detections(img, road_detections, coco_detections)
        
        # Save results
        output_path = os.path.join(PREDICTION_OUTPUT, os.path.basename(img_path))
        cv2.imwrite(output_path, result_img)
        
        # Print detection summary
        detection_counts = defaultdict(int)
        for _, label, _ in road_detections + coco_detections:
            cls = label.split()[0]
            detection_counts[cls] += 1
        
        print("Detected objects:")
        for cls, count in detection_counts.items():
            print(f"- {cls}: {count}")
        print(f"Saved to {output_path}")

def main():
    # Hardware info
    print(f"\n{'='*40}")
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*40}\n")
    
    try:
        road_model, coco_model = setup_models()
        process_images(road_model, coco_model)
        print("\n✅ All images processed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()