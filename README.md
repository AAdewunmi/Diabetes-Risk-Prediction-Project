#  Diabetes Risk Prediction: An End-to-End Machine Learning System with a Flask Web App

> **A full end-to-end machine learning and Flask web application that predicts diabetes risk and visualises explainability insights for individual or batch predictions.**
> Built with **Python, scikit-learn, pandas, SHAP**, and **Flask**, this project demonstrates both **data science excellence** and **software engineering maturity** — from raw data ingestion to interactive model deployment.
>
> ⚙️ **Two integrated components:**
>
> 1. **End-to-End Machine Learning Pipeline** — data processing → training → evaluation → explainability.
> 2. **Interactive Flask Dashboard** — real-time single & batch prediction app powered by the trained model.

---

##  Highlights

* **Automated ML Pipeline:** Modular scripts for loading, preprocessing, EDA, training, evaluation, and model explainability.
* **Interactive Web Dashboard:** Built with Flask + Bootstrap + Chart.js, for clinicians and analysts to interactively explore model predictions.
* **Explainability:** Integrated SHAP/LIME interpretability tools, visualising local and global feature contributions.
* **Production-ready structure:** Logging, testing, CI, and pre-commit hooks aligned with professional ML engineering standards.
* **Collaborative foundation:** Code documented with Javadoc-style docstrings, pytest coverage, and a clean file architecture.

---

##  Project Architecture

```
Diabetes-Risk-Prediction-Project/
├── data/                          # raw dataset (diabetes.csv)
├── models/                        # saved ML models (.joblib)
├── reports/                       # generated plots, explainability & metrics
│   ├── explain/                   # SHAP/LIME visual outputs
│   ├── models/                    # trained model artifacts
│   └── figures/                   # EDA & evaluation plots
├── src/
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── data_exploration.py
│   ├── data_visualisation.py
│   ├── statistical_analysis.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── dashboard/                 # Flask dashboard app
│       ├── app.py
│       ├── predict.py
│       ├── routes.py
│       ├── templates/
│       │   └── index.html
│       └── static/
├── tests/                         # unit, integration, and dashboard tests
├── main.py                        # unified pipeline runner
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

##  Part 1 — End-to-End Machine Learning Pipeline

This component implements a **complete data science workflow** — from ingestion to explainability — using the Pima Indians Diabetes dataset.

### ⚙️ Workflow Overview

1. **Data Loading:**
   Load `data/diabetes.csv` into pandas and validate structure.
   `python src/data_loading.py --data ./data/diabetes.csv`

2. **Data Preprocessing:**
   Handle missing values, normalize numerical features, and encode categorical variables.
   `python src/data_processing.py --data ./data/diabetes.csv --out reports`

3. **Exploratory Data Analysis (EDA):**
   Generate descriptive statistics, correlations, and visualizations (BMI, glucose, etc.).
   `python src/data_exploration.py --data ./data/diabetes.csv --out reports`

4. **Statistical Analysis:**
   Run hypothesis tests and feature significance analysis.
   `python src/statistical_analysis.py --data ./data/diabetes.csv --out reports`

5. **Model Training:**
   Train Logistic Regression, Random Forest, Gradient Boosting, or XGBoost models.
   Save the best model to `reports/models/`.

   ```bash
   python src/model_training.py --data ./data/diabetes.csv --model rf --out_dir reports
   ```

6. **Model Evaluation:**
   Evaluate accuracy, ROC AUC, and confusion matrix; save plots.
   `python src/model_evaluation.py --model reports/models/rf_best.joblib --out reports`

7. **Explainability & Feature Importance:**
   Generate SHAP plots and local explanations stored under `reports/explain/`.

8. **Run Entire Pipeline Automatically:**

   ```bash
   python main.py
   ```

---

##  Part 2 — Flask Web Application (Dashboard)

An interactive dashboard that loads the trained model from `reports/models/` and enables both **single** and **batch** predictions.

###  Quickstart (Local Run)

```bash
# 1. From the repo root
python -m pip install -r requirements.txt

# 2. Run the Flask app
PYTHONPATH=src python src/dashboard/app.py

# 3. Visit
http://127.0.0.1:5000
```

or explicitly specify a model path:

```bash
python src/dashboard/app.py --model reports/models/rf_best.joblib
```

###  Features

| Panel                       | Description                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Quick Single Prediction** | Enter medical features manually → get predicted probability and SHAP explanation.                                                 |
| **Batch CSV Upload**        | Upload a `.csv` file with multiple patients → get batch summary, visualized histogram, and downloadable explainability artifacts. |
| **Notes & Guidance**        | Practical interpretation guide for clinicians and data scientists.                                                                |

All predictions, explanations, and generated files are timestamped and stored in `reports/explain/`.

---

### Screenshots

- Single Prediction No Data

<img width="1229" height="706" alt="Image" src="https://github.com/user-attachments/assets/ba962511-77e9-408c-b625-2c30cf7da3cf" />

- Single Prediction With Data

<img width="1106" height="842" alt="Image" src="https://github.com/user-attachments/assets/208dc36f-ff58-4974-8db8-e8b3cdcd87c8" />

- Batch Prediction No Data

<img width="1197" height="482" alt="Image" src="https://github.com/user-attachments/assets/c9d97306-bf07-422f-b09d-f1f14f4639aa" />

- Batch Prediction With Data

<img width="1110" height="797" alt="Image" src="https://github.com/user-attachments/assets/bb4a45a8-7fc0-4ef7-aa83-3fc72d5d3ec4" />

---

##  Testing Strategy

Run tests locally before pushing:

```bash
PYTHONPATH=src pytest -q
```

The repository includes:

* **Unit tests:** For `ModelWrapper`, preprocessing, and data loaders.
* **API tests:** Flask routes and endpoints (`/predict`, `/predict_batch`).
* **Integration tests:** Pipeline execution to ensure end-to-end consistency.

CI/CD integration (via GitHub Actions) ensures tests run automatically on every push.

---

##  Technologies Used

* **Language:** Python 3.11+
* **Core Libraries:** pandas, numpy, scikit-learn, joblib, shap, matplotlib, seaborn
* **Web Framework:** Flask + Bootstrap + Chart.js
* **Testing:** pytest, pre-commit, black, isort, ruff
* **Tools:** VS Code, GitHub Actions CI, pre-commit hooks, reportlab for PDF export

---

##  Outputs

| Folder             | Description                             |
| ------------------ | --------------------------------------- |
| `reports/models/`  | Trained model artifacts (`.joblib`)     |
| `reports/explain/` | SHAP local & global explanations        |
| `reports/`         | EDA visuals, evaluation plots, and logs |
| `data/`            | Input dataset (`diabetes.csv`)          |
| `tests/`           | Pytest suite                            |

---

##  Deployment Notes

For production:

* Replace `app.secret_key` with an environment variable.
* Serve via **Gunicorn** or **Waitress** instead of Flask’s dev server.
* Mount static files via Nginx.
* Optionally containerize using Docker with health checks.

---

##  Example Use Case

This dashboard enables **clinicians** or **data scientists** to:

* Instantly assess diabetes risk for new patients.
* Interpret which medical features contribute most to the prediction.
* Batch-evaluate risk profiles for large datasets.
* Export explainability artifacts (HTML/PNG) for audit and reporting.

---

##  Acknowledgements

Special thanks to:

* **The National Institute of Diabetes and Digestive and Kidney Diseases** for the original dataset.
* The open-source community for continuous innovation in Python, Flask, and ML tooling.

---

##  Author

**Adrian Adewunmi**

[GitHub](https://github.com/AAdewunmi)

-----

## License

[MIT License](LICENSE)
