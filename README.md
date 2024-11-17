
Tennis Ball Detection and Player Tracking with YOLOv5 and Streamlit

This project showcases real-time detection of tennis balls and players from video inputs using a custom-trained YOLOv5 model. The interactive application is built with Streamlit, allowing users to upload videos, track detection progress, and view results seamlessly.



Key Features
- Video Upload: Easily upload `.mp4` video files for analysis.  
- Real-time Progress Updates: Track detection progress during video processing.  
- Visualization: Output video highlights detected tennis balls and players.  
- Custom YOLOv5 Model: Optimized for tennis match analysis.

---

Setup and Installation

1. Clone the Repository
```bash
git clone <repository-url>
cd yolov5
```

2. Install Dependencies
Ensure Python is installed (preferably version 3.9+). Then, install the required libraries:
```bash
pip install -r requirements.txt
```

3. Model Setup
Place your custom-trained YOLOv5 model weights file in the following path:
```
yolov5/runs/exp/weights/best.pt
```

4. Run the Application
Launch the Streamlit application locally:
```bash
streamlit run app.py
```



File Structure
```
yolov5/
├── app.py                  # Streamlit application file
├── runs/
│   └── exp/
│       └── weights/
│           └── best.pt     # Custom-trained YOLOv5 model weights
├── data/                   # Directory for input video files
└── outputs/                # Directory for saving processed output videos
```



How to Use
1. Launch the Streamlit application.
2. **Upload a Video:** Supported format is `.mp4`.
3. **Processing:** The app will analyze the video to detect tennis balls and players, showing real-time progress.
4. **Output Video:** Once complete, download or preview the video with detections highlighted.



Example Use Case
Upload a tennis match video, and the app will:
- Detect and track tennis ball movements.
- Highlight player positions.
- Generate a visual overlay showcasing the detected objects and their trajectories.



Dependencies
- **Streamlit:** For building the interactive web interface.  
- **PyTorch:** Backend library for YOLOv5 model execution.  
- **OpenCV:** For video processing and frame manipulation.  
- **YOLOv5:** State-of-the-art object detection model.  



Future Enhancements
- Live Video Detection: Extend support for live camera feeds.  
- Multi-Object Analysis: Enable simultaneous tracking of multiple tennis balls.  
- Advanced Metrics: Provide bounce detection and heatmaps for player movement.

