[English](README.md) | **Українська**
# MRI Image Analysis System for Segmenting Knee Joint Structures
## 📖 Description

This project is a software system for the automatic segmentation of anatomical structures of the knee joint in MRI images using deep learning methods.

The system allows you to:

- import medical files in MHD/RAW and DICOM formats;
- perform automatic segmentation of MRI images;
- view segmentation results in a graphical interface;
- display the patient’s DICOM metadata;
- save segmented slices in PNG format.

---

## 🚀 Key Features

✅ Medical image segmentation using U-Net
✅ Support for MHD/RAW and DICOM
✅ Visualization of segmentation masks
✅ Saving results as PNG
✅ Graphical interface built with PyQt6
✅ GPU support via CUDA

---

## 🧠 Model Architecture

The project uses the **U-Net** architecture, which consists of:

- Encoder (Downsampling)
- Bottleneck
- Decoder (Upsampling)
- Skip Connections

The following are used for training:

- Dice Loss
- Focal Loss

Evaluation metrics:

- Dice Score
- IoU (Intersection over Union)

---

# 🖥️ Requirements

## Hardware

| Component | Minimum Requirements |
|---|---|
| GPU | NVIDIA CUDA |
| VRAM | 4 GB |
| RAM | 8 GB |

---

### Software

- Python 3.11+
- CUDA Toolkit (optional)
- pip

---

## 📦 Key Libraries

- torch
- torchvision
- numpy
- pillow
- matplotlib
- pydicom
- PyQt6
## Installing Dependencies

```bash
pip install -r requirements.txt
```

<img width=48% height=48% alt="image" src="https://github.com/user-attachments/assets/ef3aa55a-cf84-4347-8d98-89736d9405df" /> <img width=48% height=48% alt="image" src="https://github.com/user-attachments/assets/05532492-72a5-4d82-b13c-32d4eb34a15b" />

