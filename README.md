# Hand and Face Mesh App

This application uses MediaPipe to detect and draw hand and face landmarks in real-time using your webcam.

## Features
- Hand landmark detection and drawing
- Face landmark detection and drawing
- Real-time processing with OpenCV

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sairam-penjarla/Face-Mesh---Mediapipe-API.git
    cd Face-Mesh---Mediapipe-API
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the MediaPipe models:
    MediaPipe will automatically download the required models when you run the application for the first time.

## Configuration
Edit the `config.yaml` file to set the desired parameters:
    ```yaml
    video_source: 0
    min_detection_confidence: 0.5
    min_tracking_confidence: 0.5
    face_min_detection_confidence: 1
    face_min_tracking_confidence: 0.5
    ```

## Usage

Run the application:
```bash
python app.py
```

Press `q` to quit the application.

## License
This project is licensed under the MIT License.
```

This refactored version of your code organizes it into a more structured format, making it easier to maintain and extend. The `HandFaceMeshApp` class encapsulates the functionality, and utility functions are moved to `utils.py`. Configuration parameters are stored in `config.yaml` for easy adjustments.