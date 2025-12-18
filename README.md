# Live Face & Emotion Analysis 🎭

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Face%20Detection-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A real-time computer vision project that detects human faces using **YOLOv8** and classifies their emotional state into 4 categories (**Angry, Happy, Sad, Surprise**) using deep learning. The system features a Tkinter-based GUI that allows users to switch between three different classification models in real-time.

---

## 🌟 Key Features

* **Real-Time Face Detection:** Uses a fine-tuned `yolov8n` model for fast and accurate face localization.
* **Multi-Model Emotion Classification:** Switch on-the-fly between three architectures:
    * **Custom CNN:** A lightweight 3-layer Convolutional Neural Network.
    * **ResNet18:** A powerful, pre-trained residual network fine-tuned for emotion.
    * **Lightweight ResNet:** A "vanilla" implementation of ResNet optimized for speed.
* **Targeted Emotion Recognition:** Specialized in distinguishing 4 specific emotions: `Angry`, `Happy`, `Sad`, and `Surprise`.
* **Interactive GUI:** A user-friendly interface built with Tkinter and OpenCV for live webcam inference.

---

## 📂 Project Structure
face-emotion-analysis/
│
├── emotion_classification/             # Core logic for emotion models
│   ├── dataset_emotion/                # Train/Test/Val and Class-wise folders for emotion training
│   ├── classification.ipynb            # Notebook for training & evaluating models
│   ├── models/                         # Model architectures
│   │   ├── cnn.py                      # Custom CNN implementation
│   │   ├── resnet18.py                 # ResNet18 wrapper
│   │   └── resnet_vanilla.py           # Lightweight ResNet implementation
│   ├── utils/                          # Utility scripts
│   │   ├── data_loader.py              # PyTorch Dataset & DataLoader
│   │   ├── train.py                    # Training loop functions
│   │   └── eval.py                     # Evaluation & Confusion Matrix
│   └── checkpoints_*/                  # Directories for saved model weights
│
├── face_detection/                     # Core logic for face detection
│   ├── dataset_face/                   # YOLO format data
│   │   ├── images/                     # .jpg images for train/val/test
│   │   └── labels/                     # .txt files with normalized YOLO bboxes
│   ├── face_detection.ipynb            # Notebook for YOLOv8 training/setup
│   └── yolo/                           # YOLO weights and configs
│
├── live_emotion_detection.ipynb        # Main application entry point (GUI)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
