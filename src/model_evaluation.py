import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(X_test, y_test):
    """
    Evaluates the trained diabetes prediction model.

    Args:
        X_test (pandas.DataFrame): The test set features.
        y_test (pandas.Series): The test set target variable.
    """
    try:
        # Load the trained model
        model = joblib.load('../models/diabetes_prediction_model.joblib')

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('../reports/confusion_matrix.png')
        plt.close()
        print("Confusion matrix plot saved to ../reports/confusion_matrix.png")

    except FileNotFoundError:
        print("Error: ../models/diabetes_prediction_model.joblib not found. Please run model_training.py first.")

if __name__ == '__main__':
    # Example usage
    try:
        evaluation_data = pd.read_csv('../data/diabetes.csv') # Directly load
        X = evaluation_data.drop(columns=['Outcome'])
        y = evaluation_data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        evaluate_model(X_test, y_test)
    except FileNotFoundError:
        print("Please ensure 'diabetes.csv' is in the data directory.")