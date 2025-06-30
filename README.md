# 🫁 Federated Lung Segmentation using DeepLabV3+

This project implements a **federated learning** pipeline for semantic lung segmentation from chest X-rays using **DeepLabV3+** with PyTorch. The segmentation is distributed across **multiple clients** without central data sharing, simulating privacy-preserving learning across hospitals or institutions.

---

## 🚀 Key Features

- Federated training with custom client update + FedAvg aggregation
- Uses DeepLabV3+ (Segmentation Models PyTorch)
- Custom Dataset + DataLoader with Albumentations
- Metric tracking: Pixel Accuracy (PA), Mean IoU (mIoU), CrossEntropy Loss
- Inference visualization of predicted vs ground truth masks

---

## 📁 Dataset

Dataset used: [Chest X-Ray Lung Segmentation Dataset](https://www.kaggle.com/datasets/andrewmvd/chest-xray-semantic-segmentation)

