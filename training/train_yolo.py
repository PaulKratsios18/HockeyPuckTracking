from pathlib import Path
import torch
import yaml

def train_yolo_model():
    # Load YOLOv5 
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Load custom configuration
    with open('training/hockey_puck.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Training settings
    hyp = {
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'fl_gamma': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0
    }
    
    # Train the model
    model.train(
        data=data_config,
        epochs=100,
        batch_size=16,
        imgsz=640,
        hyp=hyp,
        workers=8,
        project='runs/train',
        name='hockey_puck_detector'
    )

if __name__ == "__main__":
    train_yolo_model() 