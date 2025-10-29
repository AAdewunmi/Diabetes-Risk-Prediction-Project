# UNDER CONSTRUCTION

-----

# 🩺 Diabetes Risk Prediction Project: End-to-End Pipeline & Web Dashboard

This repository contains an end-to-end data science project focused on predicting the risk of diabetes from medical features. It includes the **full machine learning pipeline** (data loading, cleaning, training, evaluation, explainability) and a companion **Flask Web Dashboard** for live predictions.

## 🎯 Objective

The primary goal is to **build and deploy a robust machine learning model** (Logistic Regression, Random Forest, etc.) that accurately predicts the onset of diabetes based on medical features (e.g., Age, Glucose, BMI). The secondary goal is to serve this model via a simple, functional web interface to assist healthcare professionals in identifying high-risk individuals for early intervention.

## 💻 Overview: Pipeline & Dashboard

The project is split into two core components:

1.  **Data Science Pipeline (`src/`):** A modular workflow for data preparation, model training, evaluation, and interpretability. The master script for this is `main.py`.
2.  **Flask Dashboard (`src/dashboard/`):** A lightweight web application that loads a pre-trained model artifact and provides a user interface for real-time risk prediction.

-----

## 📂 Repository Structure

```
diabetes_risk_prediction_project/
├── data/
│   └── diabetes.csv
├── models/
│   └── diabetes_prediction_model.joblib # Trained models saved here
├── reports/
│   ├── bmi_distribution_by_outcome.png
│   ├── ... (logs, plots, evaluation results)
├── src/
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── data_exploration.py
│   ├── ... (other pipeline scripts)
│   └── dashboard/
│       ├── app.py # Flask application entry
│       └── predict.py # Model prediction logic
├── .gitignore
├── LICENCE
├── **main.py** # Runs the end-to-end ML pipeline
└── requirements.txt
```

-----

## 🛠️ Getting Started

### 1\. Initial Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AAdewunmi/Diabetes-Risk-Prediction-Project.git
    cd Diabetes-Risk-Prediction-Project
    ```
2.  **Create and activate a virtual environment (macOS/Linux):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Include SHAP/LIME if you want model explainability support
    pip install shap lime
    ```
4.  **Place the Dataset:**
    Download the `diabetes.csv` file (likely the Pima Indians Diabetes Dataset) and place it in the **`data/`** directory.

### 2\. Running the Data Science Pipeline

Execute the full pipeline to train a model and generate reports:

```bash
python main.py
```

  * This script performs all steps: data loading, processing, EDA, training (default model is Logistic Regression), and evaluation.
  * The trained model artifact (e.g., `logreg_best.joblib`) will be saved in the **`models/`** directory.
  * Plots and log files will be saved in the **`reports/`** directory.

#### Customizing Model Training

You can specify a different model for training using the `--model` CLI flag in `src/model_training.py`.

| Option | Model | Artifact Saved As |
| :--- | :--- | :--- |
| `logreg` | Logistic Regression (Default baseline) | `logreg_best.joblib` |
| `rf` | Random Forest Classifier | `rf_best.joblib` |
| `gb` | Gradient Boosting | `gb_best.joblib` |
| `xgb` | XGBoost (requires `xgboost` package) | `xgb_best.joblib` |

**Example (Training Gradient Boosting):**

```bash
python src/model_training.py --data data/diabetes.csv --model gb --out_dir reports
```

### 3\. Running the Flask Dashboard (Web App)

Once a model has been trained and saved in `models/`, you can launch the prediction dashboard:

1.  **Run locally (development):**
    ```bash
    # Ensure you are in the project root
    python src/dashboard/app.py --host 0.0.0.0 --port 5000
    ```
2.  **Access the Dashboard:** Open your browser to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

<!-- end list -->

  * By default, the dashboard tries to load the first available model artifact in the `models/` directory.
  * You can explicitly specify a model path:
    ```bash
    python src/dashboard/app.py --model models/gb_best.joblib
    ```

-----

## ⚙️ Data Science Workflow

The project follows a structured and modular workflow for end-to-end machine learning pipeline development:

### 1\. **Data Loading**

  * Source: [Kaggle – Diabetes Health Indicators Dataset (or similar)](https://www.kaggle.com/datasets/aaron7sun/diabetes-health-indicators-dataset)
  * Loaded using `pandas` via `src/data_loading.py`.

### 2\. **Data Preprocessing**

  * Missing values removed or imputed.
  * Categorical variables encoded using one-hot encoding.
  * Data normalization/standardization where appropriate.
  * Implemented in `src/data_processing.py`.

### 3\. **Exploratory Data Analysis (EDA)**

  * Exploratory data analysis (EDA) using `pandas`, `matplotlib`, and `seaborn` in `src/data_exploration.py`.
  * Extended statistical tests using `pandas`, `scipy`, and `numpy` in `src/statistical_analysis.py`.

### 4\. **Model Training**

  * Data split into training and validation sets.
  * Chosen classifier trained with cross-validation.
  * Model saved using `joblib` in the `models/` directory.
  * Handled in `src/model_training.py`.

### 5\. **Model Evaluation**

  * Evaluated using classification report, ROC AUC score, and confusion matrix.
  * Results printed and visualised in `src/model_evaluation.py`.

### 6\. **Model Explainability & Interpretability 💡**

  * **Objective:** To understand **why** the model makes a particular prediction, crucial for healthcare applications.
  * **Techniques:** **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** are used to provide global feature importance and local, per-prediction reasoning.
  * **Output:** Explanatory plots (force plots, summary plots, etc.) are generated and saved under `reports/`.

### 7\. **Feature Importance Analysis**

  * Extracted from the trained model (e.g., coefficients for LogReg, feature importances for RF/GB).
  * Top features visualized using bar plots via `src/data_visualisation.py`.

### 8\. **Data Science (End-To-End) Pipeline 🔄**

  * **Master Script:** The entire sequence of steps is executed via the central script, **`main.py`**.
  * **Automation:** Ensures **reproducibility** by running all components sequentially and generating logs.
  * **Integration:** The final saved model artifact (`models/*.joblib`) serves as the input dependency for the Flask Dashboard.

-----

## 🧪 Testing Strategy

A combination of unit and API tests ensures code reliability:

  * **Unit Tests:** For small, pure logic functions (e.g., data preprocessing helpers, model wrapper class in `src/dashboard/predict.py`).
  * **API Tests:** Using the Flask `test_client` to verify that web application endpoints respond correctly and return valid predictions.
  * **Running Tests:**
    ```bash
    pytest -q
    ```

## 🔐 Best Practices & Security Notes (Dashboard)

  * **Secrets:** Replace the placeholder `app.secret_key` with a real secret key using an environment variable in production deployments.
  * **Model Path Control:** Carefully validate the model file path to prevent arbitrary file system access.
  * **Input Validation:** Implement robust validation (e.g., using `pydantic`) on all form inputs to ensure features fall within expected clinical and statistical ranges before passing them to the model.
  * **Containerization:** A Dockerfile and healthcheck are recommended for production deployment environments (e.g., using Gunicorn/Waitress).

## 🤝 Collaboration and Contact

The project is open to contributions. Feel free to **open an issue** or submit a **pull request** for bug fixes or new features.

| Role | Author |
| :--- | :--- |
| **Author** | AAdewunmi (via GitHub: `https://github.com/AAdewunmi`) |
| **Contact** | Open an issue on the GitHub repository for questions or suggestions. |

-----


