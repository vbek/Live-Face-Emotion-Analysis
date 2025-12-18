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

```text
face-emotion-analysis/
│
├── emotion_classification/             # Core logic for emotion models
│   ├── dataset_emotion/                # Emotion dataset root
│   │   ├── train/                      # Training images (4 class folders)
│   │   ├── val/                        # Validation images (4 class folders)
│   │   └── test/                       # Testing images (4 class folders)
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
│   │   ├── images/                     # .jpg images for training/val
│   │   └── labels/                     # .txt files with normalized YOLO bboxes
│   ├── face_detection.ipynb            # Notebook for YOLOv8 training/setup
│   └── yolo/                           # YOLO weights and configs
│
├── live_emotion_detection.ipynb        # Main application entry point (GUI)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 📊 Datasets

To run the training notebooks, you will need to download the following datasets and place them in their respective folders:

1.  **Face Detection:**
    * **Location:** `face_detection/dataset_face/`
    * **Structure:** Must contain `images/` (JPG files) and `labels/` (TXT files with YOLO normalized bounding boxes).
    * **Source:** [Face Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
    * Used to train the `yolov8n-face` model.

2.  **Emotion Recognition:**
    * **Location:** `emotion_classification/dataset_emotion/`
    * **Structure:** Must contain `train/`, `val/`, and `test/` subdirectories. Inside each of these, there should be 4 folders named exactly: `Angry`, `Happy`, `Sad`, `Surprise`.
    * **Source:** [Facial Emotion Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset)
    * Used to train the CNN and ResNet classifiers.


## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/vbek/Live-Face-Emotion-Analysis.git](https://github.com/vbek/Live-Face-Emotion-Analysis.git)
    cd Live-Face-Emotion-Analysis
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Directories**
    * Ensure your datasets are placed correctly in `face_detection/dataset_face/` and `emotion_classification/dataset_emotion/`.
    * **Note:** This repository does not contain pre-trained weights. You **must** generate them by following the training steps in the **Usage** section below.

---

## 🚀 Usage

Follow these steps in the exact order to set up the system.

### Step 1: Train Face Detector (YOLO)
First, you need to fine-tune the YOLO model to detect faces.
1.  Open the notebook: `face_detection/face_detection.ipynb`
2.  Run all cells to train the model on your dataset.
3.  **Outcome:** This generates the `best.pt` weight file at `./face_detection/yolo/yolov8n_face_detection/weights/best.pt`.

### Step 2: Train Emotion Classifiers
Next, train the deep learning models to recognize the 4 emotions.
1.  Open the notebook: `emotion_classification/classification.ipynb`
2.  Run the cells to train the **CNN**, **ResNet18**, and **ResNet Vanilla** models.
3.  **Outcome:** This saves the trained `.pth` files in the `checkpoints_*` directories (e.g., `./emotion_classification/checkpoints_cnn/best.pth`).

### Step 3: Run Live Detection (GUI)
Once the models are trained and the weights exist, launch the real-time application.
```bash
# Run the notebook to start the GUI
jupyter notebook live_emotion_detection.ipynb
