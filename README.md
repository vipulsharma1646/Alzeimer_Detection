Federated Learning for Alzheimer's Disease Detection

Overview

This project decentralizes Alzheimer's Disease detection using Federated Learning (FL). It builds upon the centralized approach by Khan and Kwon (2024), who addressed class imbalance using a custom CNN on the OASIS dataset.

While the original paper relied on centralized data training, this project implements FedAvg and FedProx to achieve privacy-preserving training across decentralized clients without sharing raw MRI data.

Dataset

OASIS MRI Dataset (86,437 images)

Classes: Non Demented, Very Mild, Mild, and Moderate Dementia.

Challenge: Severe class imbalance (only 488 "Moderate" samples vs 67k "Non Demented").

Methodology

Centralized Baseline: Replicated the Khan and Kwon architecture using SGD optimizer to establish a benchmark accuracy of 99.27%.

Decentralization:

FedAvg: Implemented standard federated averaging across 10 and 20 clients.

FedProx: Applied to handle Non-IID data distribution by adding a proximal term to penalize local model drift.

Results

The decentralized FedProx model achieved accuracy comparable to the centralized baseline, proving viability for privacy-preserving medical AI.

Approach

Algorithm

Best Accuracy

Centralized

SGD (Baseline)

99.27%

Federated (10 Clients)

FedAvg

96.89%

Federated (20 Clients)

FedAvg

98.29%

Federated (20 Clients)

FedProx

98.65%

Technologies

Language: Python

Frameworks: PyTorch, Flower (flwr)

References

Khan, F. F., & Kwon, G. R. (2024). Comparison and analysis of CNN models to Address Skewed Data Issues in Alzheimer's Diagnosis.

McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data.

Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks.
