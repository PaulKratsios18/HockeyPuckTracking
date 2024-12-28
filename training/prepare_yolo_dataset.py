import os
import json
import shutil
import cv2
from sklearn.model_selection import train_test_split

def prepare_yolo_dataset():
    # Create YOLO directory structure
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in dirs:
        os.makedirs(f'yolo_dataset/{dir_path}', exist_ok=True)
    
    # Load labels
    with open('dataset/labels.json', 'r') as f:
        labels = json.load(f)
    
    # Split dataset
    images = list(labels.keys())
    train_images, val_images = train_test_split(images, test_size=0.2)
    
    # Process images and create labels
    def process_set(image_list, set_type):
        for image_name in image_list:
            # Find image in subdirectories
            found = False
            for subdir in ['hockey puck on ice', 'hockey puck close up']:
                src_path = f'dataset/hockey_pucks/{subdir}/{image_name}'
                if os.path.exists(src_path):
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find {image_name} in any subdirectory")
                continue
                
            # Copy image
            dst_path = f'yolo_dataset/images/{set_type}/{image_name}'
            shutil.copy(src_path, dst_path)
            
            # Create YOLO format label
            img = cv2.imread(src_path)
            if img is None:
                print(f"Warning: Could not read {src_path}")
                continue
                
            h, w = img.shape[:2]
            label = labels[image_name]
            
            # Convert to YOLO format (class x_center y_center width height)
            x_center = label['x'] / w
            y_center = label['y'] / h
            # Assuming puck size is roughly 3% of image width
            width = 0.03
            height = 0.03
            
            # Write label file
            label_path = f'yolo_dataset/labels/{set_type}/{os.path.splitext(image_name)[0]}.txt'
            with open(label_path, 'w') as f:
                f.write(f'0 {x_center} {y_center} {width} {height}\n')
    
    process_set(train_images, 'train')
    process_set(val_images, 'val')

if __name__ == "__main__":
    prepare_yolo_dataset() 