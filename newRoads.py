import os
import shutil
import glob
import subprocess
from ultralytics import YOLO
import torch
import time


DATASET_PATH = r"/home/heathcliff/myProject/GB Road Turns Detection.v5i.yolov8"
DATA_YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "My Test")
OUTPUT_DIR = os.path.join(DATASET_PATH, "Output")
PREDICTION_OUTPUT = os.path.join(OUTPUT_DIR, "Predictions")

def check_gpu_availability():
    """Check GPU status with more detailed info"""
    print("\n Hardware Configuration:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}") if torch.cuda.is_available() else print("CUDA: Not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}") if torch.cuda.is_available() else print("GPU: Not detected")
    print(f"RAM: {round(shutil.disk_usage('/').free / (1024**3), 1)}GB free")

def run_yolo_command(command, task_name):
    """Run command with real-time output streaming"""
    print(f"\nStarting {task_name} at {time.strftime('%H:%M:%S')}")
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
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Check return code
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

def main():
    # Verify paths
    print("\n Verifying paths...")
    for name, path in [
        ("Dataset", DATASET_PATH),
        ("Test images", TEST_IMAGES_DIR),
        ("YAML config", DATA_YAML_PATH)
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} path not found: {path}")
        print(f"✓ {name}: {path}")

    # System check
    check_gpu_availability()

    # Prepare output directory
    os.makedirs(PREDICTION_OUTPUT, exist_ok=True)

    # Training command
    train_cmd = [
        "yolo", "task=detect", "mode=train",
        "model=yolov8n.pt",
        f"data={DATA_YAML_PATH}",
        "epochs=50", "imgsz=800",
        "device=0",  # Force GPU usage
        "batch=8",   # Reduced batch size for RTX 3050
        f"project={OUTPUT_DIR}",
        "name=train",
        "exist_ok=True"
    ]

    if not run_yolo_command(train_cmd, "Training"):
        print("\nTraining failed. Cannot proceed with predictions.")
        return

    # Prediction command
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

    # Show results
    results = glob.glob(os.path.join(PREDICTION_OUTPUT, "results", "*.jpg"))
    print(f"\nFound {len(results)} predictions:")
    for img in results[:3]:  
        print(f"- {os.path.basename(img)}")

if __name__ == "__main__":
    print("🔧 Starting Road Detection Pipeline")
    try:
        main()
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
    print("\nScript completed")