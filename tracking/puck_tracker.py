from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

class PuckTracker:
    def __init__(self, model_path=None, history_points=30):
        # Initialize color detection parameters for very dark objects
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 80, 50])  # More restrictive for black
        
        # Motion history
        self.history = deque(maxlen=history_points)
        
        # Area constraints for puck (assuming puck is roughly 20-40 pixels wide)
        self.min_area = 300  # π * 10^2
        self.max_area = 1200  # π * 20^2
        
        # Circularity threshold
        self.min_circularity = 0.8  # Must be very circular
        
        # Minimum score threshold
        self.min_score = 0.7  # Higher confidence required
        
        # Debug mode
        self.debug = True
        
        # Optional ML model
        self.model = YOLO(model_path) if model_path else None
        
    def detect_puck(self, frame):
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for black objects
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        
        # More aggressive noise removal
        kernel = np.ones((5,5), np.uint8)  # Larger kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug:
            cv2.imshow('Mask', mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        debug_frame = frame.copy() if self.debug else None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
                
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if abs(1 - aspect_ratio) > 0.3:  # Must be roughly square
                continue
                
            # Calculate velocity score
            velocity_score = 1
            if len(self.history) > 0:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    last_point = self.history[-1]
                    if last_point is not None:
                        velocity = np.sqrt((cx - last_point[0])**2 + (cy - last_point[1])**2)
                        velocity_score = np.exp(-velocity / 50)  # More aggressive velocity penalty
            
            # Combined score
            score = circularity * velocity_score * (1 - abs(1 - aspect_ratio))
            
            if self.debug:
                # Draw all potential candidates in red
                cv2.drawContours(debug_frame, [contour], -1, (0, 0, 255), 2)
                cv2.putText(debug_frame, f'Score: {score:.2f}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if self.debug:
            if best_contour is not None:
                # Draw best candidate in green
                cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 2)
            cv2.imshow('Debug', debug_frame)
        
        if best_contour is not None and best_score > self.min_score:
            x, y, w, h = cv2.boundingRect(best_contour)
            center = (x + w//2, y + h//2)
            self.history.append(center)
            return np.array([x, y, x + w, y + h]), center
            
        self.history.append(None)
        return None, None 

    def track_puck(self, frame):
        # Detect puck
        bbox, center = self.detect_puck(frame)
        
        # If no detection, return None values
        if bbox is None:
            return None, None, None
        
        # Predict next position based on velocity if we have history
        prediction = None
        if len(self.history) >= 2 and self.history[-2] is not None:
            last_pos = np.array(self.history[-1])
            prev_pos = np.array(self.history[-2])
            velocity = last_pos - prev_pos
            prediction = tuple((last_pos + velocity).astype(int))
        
        return bbox, center, prediction

    def draw_tracking(self, frame, bbox, center, prediction):
        if bbox is not None:
            # Draw bounding box
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            if center is not None:
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
            
            # Draw prediction point
            if prediction is not None:
                cv2.circle(frame, prediction, 4, (255, 0, 0), -1)
                cv2.line(frame, center, prediction, (255, 0, 0), 2)
                
            # Draw motion history
            points = [p for p in self.history if p is not None]
            if len(points) >= 2:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
                
        return frame 