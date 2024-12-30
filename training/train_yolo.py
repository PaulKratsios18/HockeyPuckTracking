from ultralytics import YOLO
import os

def train_yolo_model():
    # Get absolute path to the yaml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, 'hockey_puck.yaml')
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8 nano model
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name='hockey_puck_detector'
    )

if __name__ == "__main__":
    train_yolo_model() 