from bing_images import bing
import os

def download_hockey_puck_images():
    # Create directories if they don't exist
    output_dir = "dataset/hockey_pucks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download regular hockey puck images
    bing.download_images(
        "hockey puck on ice",
        100,  # Increase this number for more images
        output_dir=output_dir,
        pool_size=20,
        file_type="jpg",
        filters='+filterui:photo-photo',  # Only real photos
        force_replace=True
    )
    
    # Download additional images with different angles
    bing.download_images(
        "hockey puck close up",
        50,
        output_dir=output_dir,
        pool_size=20,
        file_type="jpg",
        filters='+filterui:photo-photo',
        force_replace=True
    )

if __name__ == "__main__":
    download_hockey_puck_images() 