**English** | [Українська](README.uk.md)
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

<img width=48% height=48% alt="image" src="https://github.com/user-attachments/assets/7c9bd88e-6e4e-4be8-b88d-01ed19230288" />
 <img width=48% height=48% alt="image" src="https://github.com/user-attachments/assets/cf553e1c-89d1-43bc-a2ed-75098c5fd2a7" />
