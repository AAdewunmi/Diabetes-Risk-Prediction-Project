"""
Main pipeline runner for Diabetes Risk Prediction Project.

This script executes all data processing and machine learning steps in sequence,
saving logs and generated artifacts in the reports/ directory.

Each step is a standalone script inside src/ and may require specific arguments.
This pipeline ensures correct execution order and consistent output handling.

Usage:
    python main.py

Outputs:
    - Logs in reports/
    - Trained models in reports/models/
    - Evaluation artifacts in reports/eval/
    - Explainability visuals in reports/explain/

Author: (Your Name)
"""

import os
import subprocess


def run_script(script_name, args=None):
    """
    Runs a Python script from the src directory using subprocess and saves
    stdout + stderr into a log file inside reports/.

    Args:
        script_name (str): Python script filename (inside src/)
        args (list, optional): List of command-line arguments.

    Returns:
        None
    """
    if args is None:
        args = []

    # Ensure the reports directory exists
    if not os.path.exists('reports'):
        os.makedirs('reports')

    script_path = os.path.join('src', script_name)
    log_file = os.path.join('reports', f"{script_name.replace('.py', '')}_log.txt")

    with open(log_file, 'w') as log:
        print(f"\n--- Running {script_name} {' '.join(args)} ---", file=log)

        try:
            subprocess.run(
                ['python', script_path] + args,
                check=True,
                stdout=log,
                stderr=log
            )
            print(f"--- Finished running {script_name} ✅ ---", file=log)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing {script_name}: {e}", file=log)
        except FileNotFoundError:
            print(f"❌ Script not found: {script_path}", file=log)


def main():
    """
    Executes the full project pipeline in the correct order, passing required
    arguments to training/evaluation/explainability modules.
    """

    DATA_PATH = "data/diabetes.csv"
    MODEL_DIR = "reports/models"
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model_rf.joblib")

    # Ensure model output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # List: (script, argument list)
    pipeline = [
        ("data_loading.py", []),
        ("data_processing.py", []),
        ("data_exploration.py", []),
        ("data_visualisation.py", []),
        ("statistical_analysis.py", []),
        ("model_training.py",
         ["--data", DATA_PATH, "--model", "rf", "--out_dir", MODEL_DIR]),
        ("model_evaluation.py",
         ["--data", DATA_PATH, "--model_path", MODEL_PATH, "--out_dir", "reports/eval"]),
        ("model_explainability_interpretability.py",
         ["--model_path", MODEL_PATH, "--data", DATA_PATH,
          "--out_dir", "reports/explain", "--method", "all"])
    ]

    for script_name, args in pipeline:
        run_script(script_name, args)

    print("\n Pipeline completed — logs saved in reports/ ")


if __name__ == "__main__":
    main()
