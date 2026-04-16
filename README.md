# ❤️ Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning-based web application that predicts the risk of heart disease using patient health parameters. Built with Python, Scikit-Learn, and Streamlit.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 📖 About the Project

Heart disease is one of the leading causes of death globally. Early prediction and diagnosis can significantly reduce mortality rates. This project leverages machine learning algorithms to predict whether a patient is at risk of heart disease based on clinical parameters such as age, blood pressure, cholesterol levels, and more.

The system compares multiple ML models and selects the best-performing one:
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Gradient Boosting Classifier**

---

## 📊 Dataset

The project uses the **UCI Heart Disease Dataset** (Cleveland database), which contains 303 patient records with 14 attributes.

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) |
| `thal` | Thalassemia (0-3) |
| `target` | Diagnosis (1 = heart disease, 0 = no heart disease) |

---

## ✨ Features

- 🔬 **Multi-Model Comparison** — Trains and evaluates 6 different ML algorithms
- 📈 **Comprehensive EDA** — Exploratory Data Analysis with rich visualizations
- 🌐 **Interactive Web App** — Streamlit-based UI for real-time predictions
- 📊 **Performance Metrics** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🎯 **Feature Importance** — Identifies key contributing health factors
- 💾 **Model Persistence** — Saves trained models for deployment

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.8+ |
| ML Libraries | Scikit-Learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Framework | Streamlit |
| Model Saving | Joblib |

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the complete ML pipeline (training + evaluation)
```bash
python src/train_model.py
```

### Run the Exploratory Data Analysis
```bash
python src/eda.py
```

### Launch the Streamlit Web App
```bash
streamlit run src/app.py
```

The web app will open in your browser at `http://localhost:8501`

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~85% | ~86% | ~86% | ~86% |
| Random Forest | ~84% | ~84% | ~86% | ~85% |
| SVM | ~84% | ~85% | ~85% | ~85% |
| KNN | ~82% | ~83% | ~83% | ~83% |
| Decision Tree | ~78% | ~79% | ~79% | ~79% |
| Gradient Boosting | ~84% | ~85% | ~85% | ~85% |

*Note: Results may vary slightly due to random train-test splits.*

---

## 📁 Project Structure

```
heart-disease-predictor/
│
├── data/
│   └── heart.csv                 # Dataset
│
├── models/
│   └── best_model.pkl            # Saved best model
│   └── scaler.pkl                # Saved StandardScaler
│
├── outputs/
│   ├── correlation_heatmap.png   # Correlation matrix
│   ├── feature_importance.png    # Feature importance chart
│   ├── model_comparison.png      # Model accuracy comparison
│   ├── confusion_matrix.png      # Confusion matrix
│   └── roc_curve.png             # ROC curve
│
├── src/
│   ├── data_preprocessing.py     # Data loading & preprocessing
│   ├── eda.py                    # Exploratory Data Analysis
│   ├── train_model.py            # Model training & evaluation
│   └── app.py                    # Streamlit web application
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

---

## 🔮 Future Improvements

- [ ] Add more datasets for better generalization
- [ ] Implement deep learning models (ANN)
- [ ] Add patient history tracking feature
- [ ] Deploy on cloud (Heroku/AWS/GCP)
- [ ] Add API endpoints with FastAPI
- [ ] Implement cross-validation for robust evaluation

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please give it a star!**
