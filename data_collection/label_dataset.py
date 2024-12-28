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
            # Add new puck point to list
            if self.current_image:
                if self.current_image not in self.labels:
                    self.labels[self.current_image] = {"points": [], "valid": True}
                self.labels[self.current_image]["points"].append({"x": x, "y": y})
    
    def get_all_images(self):
        images = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    images.append((file, full_path))
        return images
    
    def label_images(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        images = self.get_all_images()
        if not images:
            print("No images found in the specified directories!")
            return
            
        print("Instructions:")
        print("- Click on the center of each hockey puck")
        print("- Press 'n' to move to next image")
        print("- Press 'x' to mark image as invalid (no pucks/bad image)")
        print("- Press 'c' to clear points for current image")
        print("- Press 'q' to quit and save")
        
        for image_name, image_path in images:
            self.current_image = image_name
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            while True:
                display = img.copy()
                if image_name in self.labels and self.labels[image_name]["valid"]:
                    # Draw all puck points
                    for idx, pt in enumerate(self.labels[image_name]["points"]):
                        cv2.circle(display, (pt["x"], pt["y"]), 5, (0, 255, 0), -1)
                        # Draw point number
                        cv2.putText(display, str(idx+1), 
                                  (pt["x"]+10, pt["y"]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
                elif image_name in self.labels and not self.labels[image_name]["valid"]:
                    # Show "INVALID" for marked invalid images
                    cv2.putText(display, "INVALID", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('n'):  # Next image
                    break
                elif key == ord('x'):  # Mark as invalid
                    self.labels[self.current_image] = {"points": [], "valid": False}
                elif key == ord('c'):  # Clear points
                    if self.current_image in self.labels:
                        self.labels[self.current_image] = {"points": [], "valid": True}
                elif key == ord('q'):  # Quit
                    with open('dataset/labels.json', 'w') as f:
                        json.dump(self.labels, f)
                    cv2.destroyAllWindows()
                    return
        
        with open('dataset/labels.json', 'w') as f:
            json.dump(self.labels, f)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = ImageLabeler("dataset/all_pucks")
    labeler.label_images() 