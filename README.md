# Diabetes Risk Prediction Project

This project aims to predict the risk of diabetes based on various medical measurements from the "diabetes.csv" dataset.

---

## Objective 

---

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

---

## Dataset

The dataset used for this project is `diabetes.csv`, originating from the National Institute of Diabetes and Digestive and Kidney Diseases. Please ensure this file is placed in the `data/` directory.

---

## Repository Structure

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

---

## Technologies Used: 

- Python 3.8+
- pandas, numpy
- seaborn, matplotlib
- scikit-learn, joblib
- powerpoint
- csv file
- vscode

---

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

## Documentation

-   Each Python file (`.py`) includes docstrings for functions and inline comments to explain the code.
-   This `README.md` provides an overview of the project.
-   The `requirements.txt` file lists the necessary Python packages.
-   The `reports/` directory will contain generated reports and visualizations.
-   The `reports/` directory is intended to contain the `diabetes_readmission_analysis.pptx` presentation (you will need to create this separately).

## Next Steps

-   Further explore the data in `data_exploration.py`.
-   Add more insightful visualizations in `data_visualisation.py`.
-   Perform more detailed statistical analysis in `statistical_analysis.py`.
-   Experiment with different machine learning models in `model_training.py`.
-   Enhance the model evaluation in `model_evaluation.py`.
-   Create the `diabetes_readmission_analysis.pptx` to communicate findings.
-   Create a data-driven web-application using Flask / SQL and a pipeline using Python 3. 

---

## Contact
If you have questions or suggestions, feel free to reach out or open an issue.

---

## Author

Adrian Adewunmi – [GitHub](https://github.com/AAdewunmi)
