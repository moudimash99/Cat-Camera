from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def train_model():
    # 1. Load a pre-trained model (Transfer Learning)
    # 'yolo11n.pt' is the "Nano" version (Fastest, usually good enough for cats)
    # 'yolo11s.pt' is "Small" (Slightly slower, more accurate)
    model = YOLO('yolo11s.pt') 

    # 2. Train the model
    results = model.train(
        data=r"C:\Users\Theotime\Documents\test\Cat-Camera\output\dataset_final\data.yaml", # Path to your YAML
        epochs=70,             # How many times to go through the data (50 is a good start)
        imgsz=640,             # Image resolution (640 is standard)
        batch=16,              # How many images to process at once
        device=0,              # Use '0' for GPU, 'cpu' for CPU
        name='cat_detector_v1' # Name of the output folder
    )
    
    print("Training Complete!")

if __name__ == '__main__':
    train_model()