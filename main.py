import cv2
from tracking.puck_tracker import PuckTracker

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or video file path
    
    # Initialize tracker
    tracker = PuckTracker('runs/train/hockey_puck_detector/weights/best.pt')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track puck
        bbox, center, prediction = tracker.track_puck(frame)
        
        # Draw tracking visualization
        frame = tracker.draw_tracking(frame, bbox, center, prediction)
        
        # Display frame
        cv2.imshow('Hockey Puck Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 