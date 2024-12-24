import cv2
import torch
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class PuckTracker:
    def __init__(self, model_path, history_points=30):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.5  # Detection confidence threshold
        
        # Initialize tracker
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking_active = False
        
        # Motion history
        self.history = deque(maxlen=history_points)
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        self.kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])  # State transition matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])  # Measurement function
        self.kf.R *= 10  # Measurement noise
        self.kf.Q *= 0.1  # Process noise
        
        self.prediction = None
    
    def detect_puck(self, frame):
        # Run YOLOv5 detection
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        # Filter for hockey puck class (assuming class 0)
        puck_detections = detections[detections[:, -1] == 0]
        
        if len(puck_detections) > 0:
            # Get the detection with highest confidence
            best_detection = puck_detections[np.argmax(puck_detections[:, 4])]
            return best_detection[:4]  # Return [x1, y1, x2, y2]
        return None
    
    def update_kalman(self, measurement):
        if measurement is not None:
            x, y = measurement
            if self.prediction is None:
                # Initialize Kalman filter
                self.kf.x = np.array([[x], [y], [0], [0]])
            else:
                self.kf.update(np.array([[x], [y]]))
            
        # Predict next position
        self.kf.predict()
        self.prediction = (int(self.kf.x[0]), int(self.kf.x[1]))
        return self.prediction
    
    def track_puck(self, frame):
        if not self.tracking_active:
            # Try to detect puck
            bbox = self.detect_puck(frame)
            if bbox is not None:
                self.tracker.init(frame, tuple(bbox))
                self.tracking_active = True
                center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                self.history.append(center)
                return bbox, center, self.update_kalman(center)
        else:
            # Update tracker
            success, bbox = self.tracker.update(frame)
            if success:
                center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                self.history.append(center)
                return bbox, center, self.update_kalman(center)
            else:
                self.tracking_active = False
        
        # If no detection or tracking, use Kalman prediction
        if self.prediction is not None:
            return None, None, self.prediction
        return None, None, None

    def draw_tracking(self, frame, bbox, center, prediction):
        # Draw detection/tracking box
        if bbox is not None:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Draw center point
        if center is not None:
            cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
        
        # Draw prediction
        if prediction is not None:
            cv2.circle(frame, prediction, 4, (255, 0, 0), -1)
        
        # Draw motion history
        if len(self.history) > 1:
            for i in range(1, len(self.history)):
                cv2.line(frame, 
                        (int(self.history[i-1][0]), int(self.history[i-1][1])),
                        (int(self.history[i][0]), int(self.history[i][1])),
                        (0, 255, 255), 2)
        
        return frame 