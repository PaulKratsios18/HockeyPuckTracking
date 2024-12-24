import cv2
import os
import json

class ImageLabeler:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.labels = {}
        self.current_image = None
        self.window_name = "Image Labeler"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Record center point of puck
            if self.current_image:
                self.labels[self.current_image] = {"x": x, "y": y}
    
    def label_images(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        for image_name in os.listdir(self.image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.current_image = image_name
                image_path = os.path.join(self.image_dir, image_name)
                img = cv2.imread(image_path)
                
                while True:
                    display = img.copy()
                    if image_name in self.labels:
                        pt = self.labels[image_name]
                        cv2.circle(display, (pt["x"], pt["y"]), 5, (0, 255, 0), -1)
                    
                    cv2.imshow(self.window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('n'):  # Next image
                        break
                    elif key == ord('q'):  # Quit
                        return
        
        # Save labels
        with open('dataset/labels.json', 'w') as f:
            json.dump(self.labels, f)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = ImageLabeler("dataset/hockey_pucks")
    labeler.label_images() 