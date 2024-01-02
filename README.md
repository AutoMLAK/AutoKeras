# Image Classification with On-Premise AutoML

This repository contains scripts to train and evaluate machine learning models for medical imaging tasks using AutoML approaches, specifically focusing on CNN-based architectures and AutoKeras. The training scripts allow for flexibility in terms of models and datasets, while the evaluation scripts provide metrics to gauge model performance. 
This work is related to the research pre-print - Kabilan Elangovan, Gilbert Lim, Daniel Ting et al. Medical Image Classification with On-Premise AutoML: Unveiling Insights through Comparative Analysis, 25 July 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3172493/v1]

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Pip

Before running the scripts, install all required packages:
```bash
pip install -r requirements.txt
```
## Scripts Overview

### Training Scripts

Train models using various datasets and architectures. Parameters like model type, image dimensions, and batch sizes can be specified.

- `cnn_training_script.py`: Generic script for training models with different architectures like VGG16, InceptionV3, or DenseNet.
- `autokeras_training_script.py`: Train models using the AutoKeras library with multi-GPU support.

### Evaluation Scripts

Evaluate the performance of trained models using metrics like AUC, confusion matrix, and classification reports.

- `cnn_evaluation_script.py`: Evaluate models trained with standard CNN architectures.
- `autokeras_evaluation_script.py`: Evaluate models trained using AutoKeras.

## Usage

### Training

To train a model, navigate to the directory containing the training script and run:

For generic models:

```bash
python training_script.py --train_data_dir <path> --val_data_dir <path> --test_data_dir <path>
```

For AutoKeras models (single and multi-GPU):

```bash
python autokeras_training_script.py --train_dir <path> --val_dir <path> --test_dir <path> [--use_multi_gpu]
```
### Evaluation

After training, evaluate the model using the evaluation scripts:

For generic CNN models:

```bash
python cnn_evaluation_script.py --test_data_dir <path> --model_path <path to model>
```

For AutoKeras models:

```bash
python autokeras_evaluation_script.py --test_data_dir <path> --model_path <path to model>
```

### Customization
You can customize each script's behavior through command-line arguments. For a full list of options, use the --help flag.


