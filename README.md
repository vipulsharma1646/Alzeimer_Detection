# Federated Learning Strategies for Alzheimer's Disease Detection

## Overview
[cite_start]This project performs a comparative analysis of Federated Learning (FL) strategies for detecting Alzheimer's disease using the OASIS MRI dataset[cite: 1, 2]. [cite_start]It addresses the privacy-utility trade-off in medical imaging by training models locally on decentralized data[cite: 14, 22].

## Key Features
* [cite_start]**Dataset:** OASIS MRI (86,437 images) covering four classes: Non Demented, Very Mild, Mild, and Moderate Dementia[cite: 198].
* [cite_start]**Architecture:** A lightweight, custom 4-block CNN designed for MRI feature extraction[cite: 65].
* [cite_start]**Algorithms:** Implementation and comparison of Federated Averaging (FedAvg) and FedProx[cite: 8].
* [cite_start]**Non-IID Handling:** Evaluates performance under extreme data atomization (20 clients) and class imbalance[cite: 216].

## Methodology
1.  [cite_start]**Centralized Baseline:** Established using SGD (99.27% accuracy), which outperformed Adam (97.86%)[cite: 7, 136].
2.  **Federated Setup:**
    * [cite_start]**FedAvg:** Standard weighted averaging of client parameters[cite: 76].
    * [cite_start]**FedProx:** Adds a proximal term ($\mu$) to handle statistical heterogeneity[cite: 77].

## Results
[cite_start]The study found that tuning the proximal term in FedProx recovers performance in scaled networks[cite: 11].

| Algorithm | Scenario | Accuracy |
| :--- | :--- | :--- |
| **Centralized SGD** | Baseline | [cite_start]**99.27%** [cite: 330] |
| **FedAvg** | 20 Clients | [cite_start]98.29% [cite: 330] |
| **FedProx ($\mu=0.01$)** | 20 Clients | [cite_start]**98.65%** [cite: 330] |
| **FedProx ($\mu=0.1$)** | 20 Clients | [cite_start]75.65% [cite: 330] |

## Author
[cite_start]Vipul Sharma (IISER Bhopal) [cite: 3]
