# Indoor Localization using RSSI and Deep Learning

[![DOI](https://img.shields.io/badge/DOI-10.1109/CICN67655.2025.11367998-blue)](https://doi.org/10.1109/CICN67655.2025.11367998)

Official implementation of the published IEEE conference paper:

**Anil Vansarla and Amit Agarwal**  
“AI-Based Indoor Localization Using RSSI: CNN–LSTM Based Approach,”  
*2025 IEEE 17th International Conference on Computational Intelligence and Communication Networks (CICN)*,  
Goa, India, 20–21 December 2025.  

📄 DOI: https://doi.org/10.1109/CICN67655.2025.11367998  
📚 IEEE Xplore: https://ieeexplore.ieee.org/document/11367998  

---

## Publication Details

- Conference: 2025 IEEE CICN  
- Publisher: IEEE  
- DOI: 10.1109/CICN67655.2025.11367998  
- Electronic ISBN: 979-8-3315-8733-8  
- Electronic ISSN: 2472-7555  
- Conference Location: Goa, India  
- Date of Publication on IEEE Xplore: February 2026  

---

## Overview

This project implements an indoor localization system using Bluetooth Low Energy (BLE) RSSI fingerprints and deep learning.  

The framework combines spatial feature extraction using CNNs with temporal modeling via LSTMs to improve localization accuracy in indoor environments.  

The system is packaged as a deployable machine learning service using BentoML, enabling real-time inference via a REST API.

---

## Key Contributions (as published)

- Hybrid CNN–LSTM architecture for RSSI-based indoor localization  
- Stratified 5-fold cross-validation training  
- Ensemble aggregation for improved robustness  
- Grid-level localization across 105 indoor classes  
- Production-ready deployment using BentoML  

---

## Experimental Results

The published results report **87.7% classification accuracy** using 5-fold stratified cross-validation, as documented in:

`notebooks/Indoor Localization.ipynb`

The modularized MLOps pipeline version in this repository is structured for reproducibility and deployment. Minor numerical differences may occur due to environment configuration or deterministic seed settings.

---

## Model Architecture

- **Input**: Sensor data sequences with shape `(batch_size, 20, 13)`
- **Feature Extraction**: 1D Convolutional Neural Network
- **Temporal Modeling**: Long Short-Term Memory (LSTM)
- **Ensemble Strategy**: Averaging logits from multiple fold-trained models
- **Preprocessing**: Feature scaling using fitted scalers
- **Output**: Predicted location class

Framework: **PyTorch**

---

## Project Structure

```
Indoor_localization/
├── notebooks/              # Research experiments and validation
├── src/                    # Model architecture and utilities
├── models/                 # Saved trained models
├── service.py              # BentoML service definition
├── save_model.py           # Model saving utility
├── bentofile.yaml          # BentoML configuration
├── deployment.yaml         # Deployment configuration
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/timbersaw-jugg/Indoor_localization.git
cd Indoor_localization
```

### 2️⃣ Create environment

```bash
conda create -n indoor_env python=3.10
conda activate indoor_env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Service

### Local Development

Start the BentoML service:

```bash
bentoml serve service:IndoorLocalizationClassifier --reload
```

Access interactive API interface at:

```
http://localhost:3000
```

---

## API Usage

**Endpoint**  
`POST /predict`

**Input format**

```json
{
  "input_sequence": [[[sensor_data_timestep_1], ..., [sensor_data_timestep_20]]]
}
```

**Example request**

```bash
curl -X POST "http://localhost:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{"input_sequence": [[[1,2,3,4,5,6,7,8,9,10,11,12,13], ...]]}'
```

**Response**

```json
[predicted_class_id]
```

---

## Building and Deployment

Build Bento package:

```bash
bentoml build
```

Optional containerization:

```bash
bentoml containerize <bento_tag>
```

Deploy to BentoCloud:

```bash
bentoml deploy <bento_tag> -n indoor-localization-service
```

---

## Reproducibility

- Random seed fixed during experimentation  
- 5-fold stratified cross-validation  
- Deterministic preprocessing  
- Hardware: Standard GPU training environment  

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{vansarla2025indoor,
  author={Vansarla, Anil and Agarwal, Amit},
  booktitle={2025 IEEE 17th International Conference on Computational Intelligence and Communication Networks (CICN)},
  title={AI-Based Indoor Localization Using RSSI: CNN–LSTM Based Approach},
  year={2025},
  doi={10.1109/CICN67655.2025.11367998},
  publisher={IEEE},
  address={Goa, India}
}
```

---

## License

This project is licensed under the MIT License.  
© 2025 Anil Vansarla, VNIT Nagpur

---

## Contact

**Anil Vansarla**  
Visvesvaraya National Institute of Technology (VNIT)  
GitHub: https://github.com/timbersaw-jugg  
