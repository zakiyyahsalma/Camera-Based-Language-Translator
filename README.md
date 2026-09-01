# Camera-Based Language Translator

A deep learning-based text recognition and translation system that detects text from images, recognizes the extracted text, and translates it into Indonesian.

## Overview

This project implements an end-to-end text recognition and translation pipeline using computer vision and deep learning techniques.

The system detects text regions from input images, recognizes the text content using a trained text recognition model, and automatically translates the detected text into Indonesian.

The inference pipeline combines text detection, text recognition, and machine translation to provide an automated solution for understanding foreign-language text from images.

## Features

### Text Detection

The system detects text regions from images using a document text detection model.

Features:
- Automatic text region detection
- Bounding box extraction
- Text area cropping before recognition

### Text Recognition

The system performs optical character recognition using a trained deep learning model.

Features:
- Custom trained recognition model
- Image preprocessing before inference
- Word-level text prediction

### Text Translation

The recognized text is translated automatically into Indonesian using a translation service.

Features:
- Automatic language translation
- Supports conversion of detected text into Indonesian output

### Visualization

The system provides visual output by:
- Displaying detected text regions
- Showing predicted text
- Showing translated results

## System Workflow

1. Input image is provided to the system.
2. Text regions are detected from the image.
3. Detected regions are cropped and preprocessed.
4. Recognition model predicts text from each cropped region.
5. Predicted text is combined into complete text output.
6. Recognized text is translated into Indonesian.
7. Detection results and translation output are visualized.

## Technologies Used

### Programming Language
- Python

### Deep Learning Framework
- PyTorch

### Computer Vision
- OpenCV
- NumPy
- Pillow

### OCR Framework
- docTR
- Custom text recognition model

### Translation
- Google Translate API (googletrans)

## Project Files

```
Camera-Based-Language-Translator/

├── inference.py
├── train_pytorch.py
├── utils.py
└── README.md
```

## File Description

### inference.py

Main inference pipeline responsible for:
- Loading trained recognition model
- Detecting text regions
- Performing text recognition
- Translating recognized text
- Visualizing prediction results

### train_pytorch.py

Training pipeline for the text recognition model.

Includes:
- Dataset loading
- Model training
- Validation process
- Optimization and learning rate scheduling
- Model checkpoint saving

### utils.py

Utility functions used during training.

Includes:
- Training visualization
- Learning rate monitoring
- Early stopping mechanism

## How to Run

### 1. Install Required Libraries

Install dependencies:

```
pip install torch torchvision opencv-python numpy pillow matplotlib python-doctr googletrans
```

### 2. Training Model

Run training script:

```
python train_pytorch.py
```

The training script will train the recognition model and save the model checkpoint.

### 3. Run Inference

Run inference using a trained model:

```
python inference.py --model_path PATH_TO_MODEL --image_path PATH_TO_IMAGE --arch MODEL_ARCHITECTURE --vocab VOCABULARY
```

The system will:
- Detect text regions
- Recognize text
- Translate output into Indonesian
- Display visualization results

## Model Pipeline

```
Input Image

        ↓

Text Detection

        ↓

Text Region Cropping

        ↓

Text Recognition Model

        ↓

Detected Text

        ↓

Translation

        ↓

Translated Output
```

## Future Improvements

Possible improvements:
- Deploy as a mobile application
- Improve recognition accuracy with larger datasets
- Add real-time camera translation
- Support more languages
- Optimize inference speed for edge devices


The complete documentation and experiment results are not included in this repository.

