# Diabetes Risk Prediction Project

This project aims to predict the risk of diabetes based on various medical measurements from the "diabetes.csv" dataset.


## Objective 

The goal of this project is to build a machine learning model that predicts the risk of diabetes using a dataset with medical features such as age, glucose levels, BMI, blood pressure, and others. By identifying high-risk individuals, the model can assist healthcare professionals in making early interventions.

## Overview

The project follows a standard data science workflow, with each step implemented in a separate Python script:

1.  **Data Loading:** Loads the `diabetes.csv` dataset.
2.  **Data Preprocessing:** Cleans and prepares the data for analysis.
3.  **Exploratory Data Analysis (EDA):** Explores the data through summary statistics.
4.  **Data Visualisation:** Creates informative plots to understand data patterns.
5.  **Statistical Analysis:** Performs statistical tests to gain insights.
6.  **Model Training:** Trains a logistic regression model to predict diabetes risk.
7.  **Model Evaluation:** Assesses the performance of the trained model.

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
* Loaded using `pandas` via `src/data_loader.py`.

### 2. **Data Preprocessing**

* Missing values removed or imputed.
* Categorical variables encoded using one-hot encoding.
* Data normalization/standardization where appropriate.
* Implemented in `src/preprocessing.py`.

### 3. **Exploratory Data Analysis (EDA)**

* Summary statistics, correlation heatmaps, class balance checks.
* Visualized with `matplotlib` and `seaborn` in `src/visualization.py`.

### 4. **Model Training**

* Split data into training and validation sets.
* Random Forest Classifier trained with cross-validation.
* Model saved using `joblib`.
* Handled in `src/train.py`.

### 5. **Model Evaluation**

* Evaluated using classification report, ROC AUC score, and confusion matrix.
* Results printed and visualised in `src/evaluate.py`.

### 6. **Feature Importance Analysis**

* Extracted from trained model.
* Top features visualized using bar plots.
* Found in `src/visualization.py`.

### 7. **Example Pipelines**

* `examples/run_pipeline.py`: End-to-end script to execute full pipeline.
* `examples/visualize_data.py`: Script for generating feature importance visualizations.

### 8. **Presentation**

* `presentation/diabetes_readmission_analysis.pptx`: A professional PowerPoint deck summarizing key insights for stakeholders.
## Documentation

-   Each Python file (`.py`) includes docstrings for functions and inline comments to explain the code.
-   This `README.md` provides an overview of the project.
-   The `requirements.txt` file lists the necessary Python packages.
-   The `reports/` directory will contain generated reports and visualizations.
-   The `reports/` directory is intended to contain the `diabetes_readmission_analysis.pptx` presentation (you will need to create this separately).


## Next Steps

While this project already provides a solid foundation for predicting diabetes risk, there are several ways to extend and improve it. Below are some possible future directions:

### 1. **Feature Engineering**

* **Create new features:** Based on domain knowledge, new features like interaction terms (e.g., BMI \* age) could be added.
* **Transformations:** Apply log transformations or polynomial features for certain variables to capture nonlinear relationships.

### 2. **Model Enhancements**

* **Experiment with different machine learning models:** Logistic regression is a good baseline, but more complex models such as Random Forest, Gradient Boosting, or XGBoost could be tested for improved accuracy.
* **Hyperparameter tuning:** Use grid search or random search to find the best hyperparameters for the models.
* **Cross-validation:** Implement k-fold cross-validation to ensure that the model generalizes well across different subsets of the data.

### 3. **Evaluation Metrics**

* **Confusion Matrix:** In addition to accuracy, include other metrics such as precision, recall, F1 score, ROC-AUC, etc., to get a better understanding of model performance.
* **Class Imbalance:** If the dataset is imbalanced, experiment with resampling techniques (SMOTE) or use weighted loss functions to improve model performance for the minority class.

### 4. **Deployment & Web Application**

* **Flask / FastAPI App:** Develop a simple web application where users can input their health metrics (e.g., age, glucose level) and get a real-time diabetes risk prediction.
* **Cloud Deployment:** Deploy the model using cloud platforms like AWS or Google Cloud, making the application scalable and accessible online.
* **Dockerization:** Containerize the app using Docker for easier deployment and scalability.

### 5. **Data Collection and Augmentation**

* **Acquire More Data:** A larger dataset might help in training more robust models. Collecting more diverse medical data would allow for a more generalized prediction model.
* **Synthetic Data Generation:** Use techniques like GANs (Generative Adversarial Networks) to generate synthetic medical data for training purposes.

### 6. **Explainability & Interpretability**

* **Model Explainability:** Use model interpretability tools such as SHAP or LIME to explain the predictions of the machine learning model, especially for sensitive areas like healthcare.
* **Feature Importance:** Identify which features (e.g., glucose, BMI) have the most impact on the model’s prediction to provide insights into the factors that contribute most to diabetes risk.

### 7. **Statistical Analysis**

* **Advanced Statistical Analysis:** Go beyond basic tests by applying advanced statistical techniques, such as regression analysis, ANOVA, or survival analysis, to uncover deeper insights into the relationships between variables.
* **Time-Series Analysis (if applicable):** If longitudinal data is available (e.g., measurements over time), explore time-series analysis to understand trends in diabetes risk.

### 8. **Report and Presentation**

* **Create an Interactive Dashboard:** Build an interactive dashboard using Dash or Streamlit that displays key metrics, visualizations, and predictions in real-time.
* **Final PowerPoint Presentation:** Prepare a PowerPoint presentation (e.g., `diabetes_readmission_analysis.pptx`) to summarize the project’s findings, model performance, and actionable insights for stakeholders.

### 9. **Collaboration and Open Source**

* **Invite Collaboration:** Open the project to contributions from others, including data scientists and medical experts, to refine the model and add new features.
* **Publish the Model:** Consider making the model publicly available for other researchers or healthcare providers to use, after evaluating privacy and ethical concerns.



## Contact
If you have questions or suggestions, feel free to reach out or open an issue.


## Author

Adrian Adewunmi – [GitHub](https://github.com/AAdewunmi)
