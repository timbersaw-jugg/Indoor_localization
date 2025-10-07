# Indoor Localization Classifier

A machine learning service for indoor localization using ensemble models deployed with BentoML.

This repository accompanies the research paper  
**“Indoor Localization: RSSI and Deep Learning Based Approach”**  
conducted at **Visvesvaraya National Institute of Technology (VNIT)**.

## Overview

This project implements an indoor localization system that uses sensor data to predict indoor locations. The system uses an ensemble of machine learning models for improved accuracy and is deployed as a REST API using BentoML.

## Features

- **Ensemble Model**: Combines multiple trained models for better predictions
- **REST API**: Easy-to-use HTTP API for real-time predictions
- **Scalable Deployment**: Built with BentoML for production-ready serving
- **Cross-validation**: Models trained using k-fold cross-validation

## Model Architecture

- **Input**: Sensor data sequences with shape `(batch_size, 20, 13)`
- **Models**: Ensemble of models trained on different data folds
- **Preprocessing**: Feature scaling using fitted scalers
- **Output**: Predicted location classes

## Reproducibility Note
The validated experiment achieving 87.7 % accuracy is contained in notebook section
 ['Indoor Localization.ipynb'](./notebooks/Indoor%20Localization.ipynb).
 
The Python scripts (`train_models.py`, `service.py`, etc.) are modular versions currently under refinement and may yield slightly lower accuracy (~70 %) due to refactoring.

## Project Structure

```
Indoor_localization_project/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── service.py            # BentoML service definition
├── save_model.py         # Model saving utility
├── bentofile.yaml        # BentoML configuration
└── train_models.py       # Model training script (if applicable)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Indoor_localization_project.git
   cd Indoor_localization_project
   ```

2. **Create and activate virtual environment**:
   ```bash
   conda create -n indoor_env python=3.10
   conda activate indoor_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Save your trained models** (if you have them):
   ```bash
   python save_model.py
   ```

## Usage

### Local Development

1. **Start the BentoML service**:
   ```bash
   bentoml serve service:IndoorLocalizationClassifier --reload
   ```

2. **Test the API**:
   - Open browser: http://localhost:3000
   - Use the interactive API interface
   - Send POST requests to `/predict` endpoint

### API Usage

**Endpoint**: `POST /predict`

**Input format**:
```json
{
  "input_sequence": [[[sensor_data_timestep_1], [sensor_data_timestep_2], ..., [sensor_data_timestep_20]]]
}
```

**Example request**:
```bash
curl -X POST "http://localhost:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{"input_sequence": [[[1,2,3,4,5,6,7,8,9,10,11,12,13], ...]]}'
```

**Response**:
```json
[predicted_class_id]
```

### Building and Deployment

1. **Build Bento**:
   ```bash
   bentoml build
   ```

2. **Containerize** (optional):
   ```bash
   bentoml containerize <bento_tag>
   ```

3. **Deploy to BentoCloud**:
   ```bash
   bentoml deploy <bento_tag> -n indoor-localization-service
   ```

## Model Details

- **Training**: Models trained using cross-validation
- **Scaling**: Feature scaling applied per fold
- **Ensemble**: Averaging logits from multiple models
- **Framework**: PyTorch for model implementation

## Requirements

- Python 3.10+
- PyTorch
- BentoML 1.4+
- NumPy
- Scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


This project is licensed under the [MIT License](./LICENSE) © 2025 Vansarla Anil, VNIT.


## Contact

Your Name - anil.kumar87654321@gmail.com
Project Link: https://github.com/timbersaw-jugg/Indoor_localization
