# YOLOv8m Object Detection & Tracking - Streamlit App

[**🌐 Live Demo**](https://object-detection-tracking-07.streamlit.app/)

---

## 🚀 Overview

This repository contains a **real-time object detection and tracking application** built using **YOLOv8m** and **Streamlit**. It supports multiple input sources including:

- 📷 Photo uploads  
- 🎞 Video uploads  
- 💻 Laptop webcam  
- 📱 Android IP camera (enter URL)  

The app displays **class names and object IDs with unique bounding box colors**, tracks objects across frames, and allows users to **download tracking records as JSON** for further analysis.

Additionally, a **custom YOLO training notebook** with a **ResNet50 backbone** is included for training on the **VOC2012 dataset**, featuring **custom loss functions** and **data augmentation** to improve model performance.

---

##  Features

- Real‑time object detection & tracking  
- Multi‑source input (Image/Video/Webcam/IP Camera)  
- Per‑class bounding box colors  
- Class name and object ID labels  
- Tracking history JSON file download  
- Custom YOLO training notebook (ResNet50 backbone, custom loss, augmentations)

---

##  How to use 

### 1. 📦 Clone Repository

```bash
git clone https://github.com/THE-NIKHIL07/object-detection-YOLOv8m.git
cd object-detection-YOLOv8m
