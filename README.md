Here is the concise, sticker-free version of the `README.md` based on your report.

---

# Federated Learning for Alzheimer's Disease Detection

## Project Overview

This project implements **Federated Learning (FL)** for multi-class Alzheimer's Disease detection using the OASIS MRI dataset.

The methodology builds upon the work of **Khan and Kwon (2024)**, who developed a custom CNN for centralized detection to address class imbalance. While their approach achieved high accuracy, it required centralized data aggregation.

My work extends this by decentralizing the model using **FedAvg** and **FedProx**, enabling training across 10 and 20 clients without sharing raw patient data.

## Dataset

The project uses the **OASIS MRI Dataset** (86,437 images). The data is highly imbalanced, with a severe scarcity of "Moderate Dementia" samples.

| Class Label | Image Count |
| --- | --- |
| Non Demented | 67,222 

 |
| Very Mild Dementia | 13,725 

 |
| Mild Dementia | 5,002 

 |
| Moderate Dementia | 488 

 |

## Methodology

### 1. Centralized Baseline

I replicated the high-performance baseline using a custom 4-block CNN.

* 
**Optimizer:** Stochastic Gradient Descent (SGD) was selected over Adam.


* 
**Accuracy:** Achieved **99.27%**.



### 2. Federated Implementation

I simulated a decentralized network with Non-IID data partitions across 10 and 20 clients.

* 
**FedAvg:** Evaluated as the standard FL algorithm.


* 
**FedProx:** Implemented to handle data heterogeneity by adding a proximal term (\mu) to the loss function.



## Results

The decentralized FedProx model achieved accuracy comparable to the centralized baseline.

| Scenario | Algorithm | Accuracy | Notes |
| --- | --- | --- | --- |
| **Centralized** | **SGD** | <br>**99.27%** 

 | Baseline benchmark. |
| 10 Clients | FedAvg | 96.89% 

 | High volatility due to client drift. |
| 20 Clients | FedAvg | 98.29% 

 | Slower convergence but stable. |
| **20 Clients** | **FedProx (\mu=0.01)** | <br>**98.65%** 

 | <br>**Best FL Result.** Recovered from 75% after tuning.

 |

## Tech Stack

* **Language:** Python
* 
**Frameworks:** PyTorch, Flower (flwr) 


* 
**Data:** OASIS MRI 



## References

* Khan, F. F., & Kwon, G. R. (2024). Comparison and analysis of CNN models to Address Skewed Data Issues in Alzheimer's Diagnosis.


* McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg).


* Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx).



---

**Would you like me to create the `requirements.txt` file next?**
