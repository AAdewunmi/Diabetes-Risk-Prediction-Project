# Diabetes Risk Prediction Project

This project aims to predict the risk of diabetes based on various medical measurements from the "diabetes.csv" dataset.


## Objective 

The goal of this project is to build a machine learning model that predicts the risk of diabetes using a dataset with medical features such as age, glucose levels, BMI, blood pressure, and others. By identifying high-risk individuals, the model can assist healthcare professionals in making early interventions.


## Overview

The project follows a standard data science workflow, with each step implemented in a separate Python script:

1.  **Data Loading:** Loads the `diabetes.csv` dataset.
2.  **Data Preprocessing:** Cleans and prepares the data for analysis.
3.  **Exploratory Data Analysis (EDA):** Explores the data through summary statistics.
6.  **Model Training:** Trains a logistic regression model to predict diabetes risk.
7.  **Model Evaluation:** Assesses the performance of the trained model.
6.  **Model Explainability & Interpretability:** 
Use model interpretability tools such as SHAP or LIME to explain the predictions of the machine learning model, especially for sensitive areas like healthcare.
7.  **Feature Importance Analysis:** 

7.  **Data Science (End-To-End) Pipeline:** 

4.  **Data Visualisation:** Creates informative plots to understand data patterns.
5.  **Statistical Analysis:** Performs statistical tests to gain insights.


You can run all these steps using the `main.py` script.


## Dataset

The dataset used for this project is `diabetes.csv`, originating from the National Institute of Diabetes and Digestive and Kidney Diseases. Please ensure this file is placed in the `data/` directory.


## Repository Structure

```
diabetes_risk_prediction_project/
├── data/
│   └── diabetes.csv
├── models/
│   └── diabetes_prediction_model.joblib
├── reports/
│   ├── bmi_distribution_by_outcome.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── data_exploration_log.txt
│   ├── data_loading_log.txt
│   ├── data_processing_log.txt
│   ├── data_visualisation_log.txt
│   ├── eda_correlation_bmi_glucose_bp.png
│   ├── eda_glucose_vs_age_outcome.png
│   ├── eda_insulin_by_outcome_boxplot.png
│   ├── model_evaluation_log.txt
│   ├── model_training_log.txt
│   ├── outcome_distribution.png
│   ├── pairplot_selected_features.png
│   └── statistical_analysis_log.txt
├── src/
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── data_exploration.py
│   ├── data_visualisation.py
│   ├── statistical_analysis.py
│   ├── model_training.py
│   └── model_evaluation.py
├── .gitignore
├── LICENCE
├── main.py
├── README.md
└── requirements.txt
```


## Technologies Used: 

- Python 3.8+
- pandas, numpy
- seaborn, matplotlib
- scikit-learn, joblib
- powerpoint
- csv file
- vscode


## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AAdewunmi/Diabetes-Risk-Prediction-Project.git]
    cd diabetes_risk_prediction_project
    ```
2. **Create a virtual environment (macOS/Linux)**

    ```bash
    # Set up virtual environment
    python3 -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the dataset:** Download the `diabetes_data.csv` file from Kaggle and place it in the `data/` directory.

5.  **Run the analysis:** Execute the `main.py` script to run all steps:
    ```bash
    python main.py
    ```


## Workflow

The project follows a structured and modular workflow for end-to-end machine learning pipeline development:

### 1. **Data Loading**

* Source: [Kaggle – Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/aaron7sun/diabetes-health-indicators-dataset)
* Loaded using `pandas` via `src/data_loading.py`.

### 2. **Data Preprocessing**

* Missing values removed or imputed.
* Categorical variables encoded using one-hot encoding.
* Data normalization/standardization where appropriate.
* Implemented in `src/data_preprocessing.py`.

### 3. **Exploratory Data Analysis (EDA)**

* Exploratory data analysis (EDA) on a diabetes dataset with `pandas` and `matplotlib` and `seaborn` in `src/data_exploration.py`.

* Extended statistical tests on the diabetes dataset with `pandas`, `scipy` and `numpy` in `src/statistical_analysis.py`.

### 4. **Model Training**

* Split data into training and validation sets.
* Random Forest Classifier trained with cross-validation.
* Model saved using `joblib`.
* Handled in `src/model_training.py`.

### 5. **Model Evaluation**

* Evaluated using classification report, ROC AUC score, and confusion matrix.
* Results printed and visualised in `src/model_evaluation.py`.

### 6. **Model Explainability & Interpretability**

#TODO

### 7. **Feature Importance Analysis**

* Extracted from trained model.
* Top features visualized using bar plots.
* Found in `src/data_visualisation.py`.

### 8. **Data Science (End-To-End) Pipeline**

#TODO

### 9. **Documentation**

-   Each Python file (`.py`) includes docstrings for functions and inline comments to explain the code.
-   This `README.md` provides an overview of the project.
-   The `requirements.txt` file lists the necessary Python packages.
-   The `reports/` directory will contain generated reports and visualizations.


### 10. **Collaboration and Open Source**

* **Invite Collaboration:** Open the project to contributions from others, including data scientists and medical experts, to refine the model and add new features.
* **Publish the Model:** Consider making the model publicly available for other researchers or healthcare providers to use, after evaluating privacy and ethical concerns.


### 11. **Contact**
If you have questions or suggestions, feel free to reach out or open an issue.


### 12. **Author**

Adrian Adewunmi – [GitHub](https://github.com/AAdewunmi)
