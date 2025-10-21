# Driver Drowsiness Detection System 😴🚗

A literature-based system developed as a final project for a Deep Learning course (CS 570). This project utilizes Computer Vision and Deep Learning techniques to monitor a driver's state and detect signs of drowsiness in real-time.

---

## 💡 Project Overview

The primary goal of this project is to enhance road safety by proactively identifying driver fatigue. The system works by analyzing video feeds to classify the state of the driver's eyes, determining if they are **open** or **closed**.

### Key Features
* **Real-time Eye State Classification:** Uses a pre-trained CNN model for high-speed eye state classification.
* **Region of Interest (ROI) Detection:** Employs **Haar Cascade Classifiers** to accurately detect the face and eyes, localizing the region of interest for analysis.
* **Drowsiness Alert:** Drowsiness is inferred and an alert is triggered based on the frequency and duration of closed eyes (i.e., counting the number of "eye flips" or blinks within a specific time frame).

---

## 🛠️ Technology Stack

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Core Model** | **Convolutional Neural Network (CNN)** | Eye state classification (Open/Closed). |
| **Object Detection** | **Haar Cascade Classifiers** | Face and eye detection for ROI segmentation. |
| **Development Language** | **Python** (primarily) | Core logic and scripting. |
| **Exploration/Training** | **Jupyter Notebook** | Model training, visualization, and experimentation. |
| **Other Libraries** | *OpenCV*, *TensorFlow/Keras*, *Dlib* (Likely) | Video stream handling, deep learning framework, facial landmark detection (Hypothetical but common). |

---

## 🚀 Getting Started

To run this project locally, follow these steps.

### Prerequisites

Ensure you have Python (3.x recommended) installed.

```bash
# Recommended environment setup
conda create -n ddd_env python=3.9
conda activate ddd_env
```

## Installation
### Clone the repository:

```
  git clone [https://github.com/NipuniSenani/DeepLearning-FinalProject.git](https://github.com/NipuniSenani/DeepLearning-FinalProject.git)
cd DeepLearning-FinalProject
```

### Install dependencies: (Note: Since the detailed requirements.txt is not provided, this is a provisional list based on project type.)
```
  pip install numpy opencv-python tensorflow keras
# You may also need dlib for advanced facial feature detection
# pip install dlib
```

### Repository Structure
The repository contains the following major directories and files:
```
DeepLearning-FinalProject/
├── Final_Project/             # Main source code, model files, and implementation scripts.
├── CS 570 Final project proposal (1).pdf  # Official project proposal document.
├── CS570_final presentation (1).pdf      # Final project presentation slides.
├── README.md                  # This README file.
└── .DS_Store
```


#### [Project Proposal](https://github.com/NipuniSenani/CS570-Deep-Learning-F23/blob/1574f1521cf7edb866d055a2d35ca4091942ddeb/Final%20Project/CS%20570%20Final%20project%20proposal%20(1).pdf)

#### [Final Presentation](https://github.com/NipuniSenani/CS570-Deep-Learning-F23/blob/1574f1521cf7edb866d055a2d35ca4091942ddeb/Final%20Project/CS570_final%20presentation%20(1).pdf)



## Outputs & Visuals
The system provides real-time detection, highlighting the driver and their eyes.

Example of System Output (Drowsiness Detected): Drowsiness is detected by counting the number of eye flips (blinks) in a given time period.

Drowsiness detect by counting number of eye flips in a given time period

![Screenshot 2024-02-25 at 5 37 05 PM](https://github.com/NipuniSenani/CS570-Deep-Learning-F23/assets/81766272/3f8aa65b-9b62-47a9-a7d0-b7a7c28bb14f)


* Detecting open eyes
  
![Screenshot 2024-02-25 at 5 37 35 PM](https://github.com/NipuniSenani/CS570-Deep-Learning-F23/assets/81766272/fce40663-6a69-42f9-a874-79bba9f7dba1)

* Detecting close eyes
  
![Screenshot 2024-02-25 at 5 37 19 PM](https://github.com/NipuniSenani/CS570-Deep-Learning-F23/assets/81766272/65a9c68b-b7a7-464a-8e8b-aaff9d44e8ae)


## Contributing
This project was developed as a final academic work. While general contributions may not be sought, bug reports and suggestions are welcome


