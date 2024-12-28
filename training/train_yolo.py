from ultralytics import YOLO

def train_yolo_model():
    # Load a model
    model = YOLO('yolov5s.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data='training/hockey_puck.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='hockey_puck_detector'
    )

if __name__ == "__main__":
    train_yolo_model() 