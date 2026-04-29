# Breast Cancer Detection Project

An AI-powered machine learning system for detecting breast cancer using clinical tabular data. This project implements multiple classification algorithms with comprehensive model evaluation, comparison, and real-time prediction capabilities.

## 🎯 Project Overview

This project develops and evaluates multiple machine learning models to predict breast cancer diagnosis from clinical features. It includes data preprocessing, model training, hyperparameter tuning, performance evaluation with ROC-AUC analysis, and a deployment-ready prediction interface.

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset with 30 clinical features derived from cell nuclei measurements.

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and preprocessed data
│   └── input/
│       └── patients.csv        # Patient data for predictions
├── notebooks/
│   └── 01_tabular_baseline.ipynb    # EDA and baseline model
├── src/
│   ├── train.py                # Model training pipeline
│   ├── evaluate.py             # Model evaluation metrics
│   ├── model.py                # Model definitions and utilities
│   ├── data_preprocessing.py    # Data cleaning and feature engineering
│   ├── predict_csv.py          # Batch prediction from CSV
│   ├── manual_predict_dynamic.py # Interactive predictions
│   ├── models/                 # Pre-trained model files
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
│   │   ├── k-nearest_neighbors.joblib
│   │   ├── support_vector_machine_(svm).joblib
│   │   └── neural_network.joblib
│   ├── scaler.joblib           # Fitted feature scaler
│   └── best_model.joblib       # Best performing model
├── reports/
│   ├── figures/                # Visualization outputs
│   └── model_metrics_summary.csv # Performance comparison
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd breast-cancer-project
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Train Models
```bash
python src/train.py
```

### Evaluate Models
```bash
python src/evaluate.py
```

### Make Predictions
```bash
# Batch predictions from CSV
python src/predict_csv.py --input data/input/patients.csv

# Interactive prediction
python src/manual_predict_dynamic.py
```

### Explore Analysis
```bash
jupyter notebook notebooks/01_tabular_baseline.ipynb
```

## 📊 Models Implemented

| Model | Training Time | Accuracy | ROC-AUC |
|-------|---------------|----------|---------|
| Logistic Regression | Fast | High | Good |
| Decision Tree | Very Fast | High | Good |
| Random Forest | Moderate | Very High | Excellent |
| K-Nearest Neighbors | Fast | High | Good |
| Support Vector Machine | Moderate | Very High | Excellent |
| Neural Network | Slow | Very High | Excellent |

*See `model_metrics_summary.csv` for detailed performance metrics*

## 📈 Features

- **Data Preprocessing**: Standardization, missing value handling, and feature scaling
- **Multiple Algorithms**: 6 different ML algorithms for comparison
- **Model Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC curves
- **Hyperparameter Tuning**: Optimized parameters for each model
- **Batch Predictions**: Process multiple patient records from CSV
- **Real-time Predictions**: Interactive prediction interface
- **Model Persistence**: Trained models saved for deployment
- **Visualization**: ROC curves, confusion matrices, and performance plots

## 📋 Dataset Information

**Source:** UCI Machine Learning Repository - Wisconsin Diagnostic Breast Cancer (WDBC)

**Samples:** 569 patient records
**Features:** 30 computed features from cell nuclei measurements
**Target:** Binary classification (Malignant/Benign)
**Features Include:**
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Symmetry, fractal dimension
- And their mean, standard error, and worst values

## 🔍 Key Files

- `src/train.py` - Training pipeline for all models
- `src/evaluate.py` - Model performance evaluation
- `src/model.py` - Model definitions and utilities
- `src/data_preprocessing.py` - Data cleaning and transformation
- `full_dataset.csv` - Complete dataset used
- `model_metrics_summary.csv` - Comparative metrics

## 📊 Performance & Results

All models achieve high accuracy (>95%) on the test set. The best performer is stored in `src/best_model.joblib`.

View detailed metrics and visualizations in:
- `reports/model_metrics_summary.csv`
- `reports/figures/` (ROC curves, confusion matrices)

## 🛠️ Usage Examples

### Training a single model:
```python
from src.model import train_model
from src.data_preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()
model = train_model(X_train, y_train, model_type='random_forest')
```

### Making predictions:
```python
import joblib
model = joblib.load('src/best_model.joblib')
predictions = model.predict(X_test)
```

## 📝 Requirements

See `requirements.txt` for all dependencies:
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ✉️ Contact & Support

For questions, issues, or suggestions, please open an issue on GitHub.

## 📚 References

- UCI ML Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn Documentation: https://scikit-learn.org/
- Breast Cancer Detection Research

---

**Status:** ✅ Active
**Last Updated:** April 2026
