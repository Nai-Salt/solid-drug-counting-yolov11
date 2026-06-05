# YOLOv11-Based Real-Time Solid Drug Detection and Counting

This repository contains the source code, sample data, training results, and supplementary materials for the paper:

**Comparison of YOLOv11n and YOLOv11s for Real-Time Solid Drug Detection and Counting**

The project implements a webcam-based solid drug detection and counting system using YOLOv11 and OpenCV. The system detects and counts two visual classes of solid drugs: **large round tablets** and **small round tablets**.

---

## Project Overview

Manual drug counting is a repetitive task in pharmacy workflows and may be affected by fatigue, time pressure, and concentration level. This project explores a vision-based counting approach using object detection to support solid drug counting.

The proposed system is designed as a research prototype. It is not intended to replace pharmacist verification or to serve as a medical-grade dispensing device.

---

## Main Features

- Real-time solid drug detection using YOLOv11.
- Webcam-based image acquisition using OpenCV.
- Detection of two tablet classes:
  - `Large_Round`
  - `Small_Round`
- Bounding box-based counting logic.
- Comparison between YOLOv11n and YOLOv11s.
- Evaluation using precision, recall, mAP50, mAP50-95, validation loss, training time, and counting accuracy.

---

## Dataset Classes

The dataset consists of two visual classes:

| Class ID | Class Name | Description |
|---|---|---|
| 0 | Large_Round | Large round tablet |
| 1 | Small_Round | Small round tablet |

The dataset was annotated using bounding boxes in YOLO format.

---

## Repository Structure

```text
solid-drug-counting-yolov11/
│
├── README.md
├── webcam_counter.py
├── training_yolov11.py
├── best.pt
├── data.yaml
│
├── results/
│   ├── results_nano.csv
│   ├── results_small.csv
│   ├── confusion_matrix_nano.png
│   ├── confusion_matrix_small.png
│   ├── result graphic_nano.png
│   └── result graphic_small.png
│
└── sample_images/
    ├── sample_1.jpg
    ├── sample_2.jpg
    └── sample_3.jpg
