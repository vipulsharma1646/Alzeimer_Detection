# Federated Learning Strategies for Alzheimer's Disease Detection

## Overview
This project performs a comparative analysis of Federated Learning (FL) strategies for detecting Alzheimer's disease using the OASIS MRI dataset It addresses the privacy-utility trade-off in medical imaging by training models locally on decentralized data[cite: 14, 22].

## Key Features
**Dataset:** OASIS MRI (86,437 images) covering four classes: Non Demented, Very Mild, Mild, and Moderate Dementia[cite: 198].
**Architecture:** A lightweight, custom 4-block CNN designed for MRI feature extraction[cite: 65].
**Algorithms:** Implementation and comparison of Federated Averaging (FedAvg) and FedProx[cite: 8].
**Non-IID Handling:** Evaluates performance under extreme data atomization (20 clients) and class imbalance[cite: 216].

## Methodology
1.**Centralized Baseline:** Established using SGD (99.27% accuracy), which outperformed Adam (97.86%)[cite: 7, 136].
2.  **Federated Setup:**
    **FedAvg:** Standard weighted averaging of client parameters[cite: 76].
    **FedProx:** Adds a proximal term ($\mu$) to handle statistical heterogeneity[cite: 77].

## Results
The study found that tuning the proximal term in FedProx recovers performance in scaled networks[cite: 11].

| Algorithm | Scenario | Accuracy |
| :--- | :--- | :--- |
| **Centralized SGD** | Baseline | **99.27%**  |
| **FedAvg** | 20 Clients | 98.29%  |
| **FedProx ($\mu=0.01$)** | 20 Clients | **98.65%**  |
| **FedProx ($\mu=0.1$)** | 20 Clients | 75.65%  |

## Author
Vipul Sharma (IISER Bhopal) 
