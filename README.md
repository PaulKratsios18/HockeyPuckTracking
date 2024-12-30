# Hockey Puck Tracking

## Overview

The Hockey Puck Tracking project is designed to track a hockey puck in real-time using computer vision techniques. This project leverages the YOLOv8 model for object detection and OpenCV for video processing, providing a robust solution for tracking a puck during stickhandling exercises.

## Features

- **Real-time Puck Detection**: Utilizes YOLOv8 for accurate puck detection.
- **Motion Tracking**: Tracks the puck's movement across frames, predicting its future position.
- **Visualization**: Displays bounding boxes, center points, and predicted paths for the puck.
- **Customizable Parameters**: Allows tuning of detection thresholds and tracking parameters for optimal performance.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/HockeyPuckTracking.git
   cd HockeyPuckTracking
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 Weights**:
   Download the pre-trained YOLOv8 weights and place them in the `runs/detect/hockey_puck_detector6/weights/` directory.

## Usage

1. **Collect and Label Data**:
   - Use `data_collection/collect_images.py` to gather images of hockey pucks.
   - Label the images using `data_collection/label_dataset.py`.

2. **Prepare Dataset for YOLO**:
   - Run `training/prepare_yolo_dataset.py` to organize the dataset for training.

3. **Train the YOLO Model**:
   - Use `training/train_yolo.py` to train the model on your dataset.

4. **Run the Tracker**:
   - Execute `main.py` to start the real-time puck tracking using your webcam or a video file.

## Configuration

- **YOLO Configuration**: Modify `training/hockey_puck.yaml` to adjust dataset paths and class names.
- **Tracking Parameters**: Adjust parameters in `tracking/puck_tracker.py` to fine-tune detection and tracking performance.

## Code Structure

- **`main.py`**: Entry point for running the puck tracking application.
- **`tracking/puck_tracker.py`**: Contains the `PuckTracker` class for detecting and tracking the puck.
- **`training/`**: Scripts for preparing the dataset and training the YOLO model.
- **`data_collection/`**: Scripts for collecting and labeling images.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of YOLO and OpenCV for their powerful tools.