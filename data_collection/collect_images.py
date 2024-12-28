from bing_image_downloader import downloader as bing
import os
import shutil
import glob

def download_hockey_puck_images():
    # Create directories if they don't exist
    output_dir = "dataset/hockey_pucks"
    temp_dir = "dataset/temp_downloads"
    final_dir = "dataset/all_pucks"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    # List of search queries
    queries = [
        "hockey puck on ice",
        "hockey puck close up",
        "hockey puck",
        "hockey puck in driveway",
        "hockey puck on grass",
        "hockey puck in backyard",
        "hockey puck on synthetic ice"
    ]
    
    # Download images for each query to temporary directories
    for idx, query in enumerate(queries):
        query_dir = os.path.join(temp_dir, f"query_{idx}")
        bing.download(
            query,
            limit=100,
            output_dir=temp_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )
    
    # Aggregate all images with unique names
    file_counter = 0
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                # Create new filename with counter
                ext = os.path.splitext(file)[1]
                new_filename = f"puck_{file_counter:04d}{ext}"
                dst_path = os.path.join(final_dir, new_filename)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    file_counter += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    print(f"Successfully collected {file_counter} unique images")

if __name__ == "__main__":
    download_hockey_puck_images() 