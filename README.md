# Alzeimer_Detection

# Comparative Analysis of Federated Learning Strategies for Alzheimer's Disease Detection

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Framework](https://img.shields.io/badge/Framework-Flower-orange)
![Library](https://img.shields.io/badge/Library-PyTorch-red)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-blue)

## 📌 Project Overview

This project explores the application of **Federated Learning (FL)** for detecting Alzheimer's disease using the **OASIS MRI dataset**. The study addresses the critical "Privacy-Utility Trade-off" in medical imaging, where strict regulations (HIPAA/GDPR) prevent the centralization of sensitive patient data.

We implemented and benchmarked two federated algorithms—**Federated Averaging (FedAvg)** and **FedProx**—against a robust centralized baseline. The project specifically focuses on optimization stability under **Non-IID (Non-Independent and Identically Distributed)** conditions and extreme data atomization (20+ clients).

## 📊 Key Results

The study established that while FedAvg performs well in smaller networks, **FedProx** is superior for larger, heterogeneous networks when the proximal term ($\mu$) is properly tuned.

| Scenario | Algorithm | Configuration | Final Accuracy |
| :--- | :--- | :--- | :--- |
| **Centralized Baseline** | SGD | LR=0.001, Momentum=0.9 | **99.27%** |
| **10 Clients** | FedAvg | Standard | 96.89% |
| **10 Clients** | FedProx | $\mu=0.1$ | 97.71% |
| **20 Clients** | FedAvg | Standard | 98.29% |
| **20 Clients** | FedProx | $\mu=0.1$ (Untuned) | 75.65% |
| **20 Clients** | **FedProx** | **$\mu=0.01$ (Tuned)** | **98.65%** |

## 🧠 Model Architecture

We designed a lightweight, custom **4-block Convolutional Neural Network (CNN)** optimized for MRI feature extraction while minimizing computational overhead for local clients.

* **Input:** $128 \times 128 \times 3$ images
* **Feature Extraction:** 4 Blocks utilizing Conv2D, Batch Normalization, ReLU, and MaxPool.
* **Regularization:** Dropout ($p=0.3$, $p=0.4$) used to strictly control overfitting on small local datasets.
* **Classifier Head:** Global Average Pooling $\rightarrow$ Dense(128) $\rightarrow$ Softmax (4 classes).

## 📂 Dataset

The project utilizes the **Open Access Series of Imaging Studies (OASIS)** MRI dataset, comprising **86,437 images**. The dataset is highly imbalanced, creating a significant challenge for federated optimization:

* **Non Demented:** ~67,000 images
* **Very Mild Dementia:** ~13,700 images
* **Mild Dementia:** ~5,000 images
* **Moderate Dementia:** ~488 images (Critical Scarcity)

## 🛠️ Methodology

1.  **Centralized Baseline:** Established a performance ceiling (99.27%) using SGD, proving it superior to Adam for this specific topography.
2.  **Federated Simulation:**
    * **FedAvg:** Standard weighted averaging of client parameters.
    * **FedProx:** Adds a proximal term $\frac{\mu}{2}||w-w^t||^2$ to the local loss function to penalize updates that drift too far from the global model.
3.  **Non-IID Simulation:** Data was randomly split across 10 and 20 clients, creating extreme data atomization where some clients received as few as 20 examples of the "Moderate Dementia" class.

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch (GPU support recommended)
* Flower (flwr)
* NumPy, Pandas, Matplotlib

### Installation
```bash
git clone [https://github.com/vipulsharma1646/Alzeimer_Detection.git](https://github.com/vipulsharma1646/Alzeimer_Detection.git)
cd Alzeimer_Detection
pip install -r requirements.txt
