# Real-Time-3D-Human-Pose-Estimation-for-Ergonomics-using-YOLOv8

Author: Aadesh Vijayaraghavan
Current Program: M.Sc. in Data Science, University of Guelph, Canada
Undergraduate: B.E. in Computer Science and Engineering, Sri Venkateswara College of Engineering (SVCE), India

Project Overview
This project presents a Real-Time 3D Human Pose Estimation (3D-HPE) system that detects and corrects poor sitting postures using deep learning and computer vision.
The model is designed to prevent workplace-related health issues such as:
Text Neck Syndrome
Musculoskeletal Disorders (MSD)
Carpal Tunnel Syndrome
Computer Vision Syndrome
By combining YOLOv8x-pose and MediaPipe, the system identifies posture deviations through a live webcam feed and triggers on-screen alerts when a user exhibits prolonged poor posture.

Objective
To apply deep-learning-based 3D pose estimation to improve ergonomic health through posture awareness and correction.
Motivation
With increased digital screen exposure in work-from-home and office environments, maintaining correct posture has become essential.
This system acts as a personal ergonomic assistant, detecting bad posture in real time and helping users adjust before long-term health effects occur.

Technologies Used
Category	Tools & Libraries
Programming	Python
Deep Learning	YOLOv8, PyTorch, MediaPipe
Model Files	.pt, .sav
Computer Vision	OpenCV
Data Handling	NumPy, Pandas
Visualization	Matplotlib
Deployment (Optional)	Flask / Streamlit

Folder Structure
RealTime-3D-HumanPose-YOLOv8
│
├── miniprojectfinal.py          # Main project script
├── model testing.py             # Testing and evaluation
├── yolo.py                      # Helper functions for YOLO model
│
├── yolov8n.pt                   # YOLOv8 base model
├── yolov8n-cls.pt               # Classification model
├── yolov8n-pose.pt              # Pose estimation weights
├── yolov8x-pose-p6.pt           # High-accuracy pretrained weights
│
├── model.sav                    # Serialized trained model
│
├── datasets/                    # Dataset and annotation files
│   └── ...                         # COCO-Pose or custom data
│
├── runs/                        # YOLOv8 training results and logs
│   └── detect/pose/                # Metrics, model checkpoints
│
└── README.md                    # Project documentation

Methodology
1.Data Collection
Captures 2D video input from a webcam.
Annotated frames used for pose-keypoint detection.
Built on COCO-Pose dataset supporting 17 keypoints.
2.Object Detection
Detects the person using YOLOv8 with bounding boxes.
Ensures real-time inference at frame-level granularity.
3.Keypoint Estimation
Extracts major joints: head, neck, shoulders, eyes, ears.
Computes inclination and curvature to determine posture quality.
4.Pose Estimation
Fine-tuned YOLOv8x-pose model trained for 100 epochs using the Adam optimizer and ReLU activation.
Achieved mAP = 91.2 % for accuracy and IoU-based precision.
5.Alert Notification
Detects “bad posture” conditions in real time.
Displays alert:
Please Sit Straight to improve posture.

Results
Metric	Model Variant	Value
Mean Average Precision (mAP)	YOLOv8x-pose	91.2 %
Optimizer	Adam	
Activation	ReLU	
Epochs	100	
Dataset	COCO-Pose (17 Keypoints)	
Real-time inference achieved
Robust to lighting, background, and angle variations

Key Contributions
Developed a 3D Human Pose Estimation model for ergonomic monitoring.
Implemented YOLOv8x-pose with high-accuracy 17-keypoint mapping.
Created an automated feedback system for posture correction.
Demonstrated deployment capability for real-time webcam applications.

Future Scope
Extend to multi-person tracking in collaborative workspaces.
Build browser or mobile extensions for desk ergonomics.
Integrate IoT wearables for long-term ergonomic analytics.
Add Power BI / Streamlit dashboards for usage insights.

About the Author
Aadesh Vijayaraghavan
Master in Data Science, University of Guelph, Canada
B.E. in Computer Science, Sri Venkateswara College of Engineering (SVCE), India
Areas of Interest: Machine Learning • Computer Vision • AI for Human Well-Being • Data Analytics
