import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(X_test, y_test):
    """
    Evaluates the trained diabetes prediction model using the test set,
    providing context for each metric.

    This function loads the trained model, makes predictions, and then
    calculates and displays key evaluation metrics such as accuracy,
    classification report, and confusion matrix. It also saves the
    confusion matrix plot in the 'reports' directory.

    Args:
        X_test (pandas.DataFrame): The test set features.
        y_test (pandas.Series): The test set target variable
        (diabetes outcome).

    Returns:
        None

    Example:
        # Assuming X_test and y_test are prepared
        evaluate_model(X_test, y_test)
    """
    try:
        # Load the trained model
        model_path = 'models/diabetes_prediction_model.joblib'
        model = joblib.load(model_path)
        print(f"\nLoaded trained model from: {model_path}")

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        print("\n--- Model Evaluation ---")

        # Accuracy: Proportion of correct predictions
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(
            "Accuracy represents the proportion of correctly classified "
            "instances out of the total instances in the test set."
        )
        print(
            (
                (
                    (
                        "In this case, the model correctly predicted the "
                        "diabetes outcome for "
                        f"{accuracy * 100:.2f}% of the patients in the test "
                        " set."
                    )
                )
            )
        )

        # Classification Report: Precision, Recall, F1-score for each class
        print("\nClassification Report:")
        report = classification_report(
            y_test, y_pred, target_names=['No Diabetes', 'Diabetes']
        )
        print(report)

        print("\nBreakdown of the classification report:")
        print(
            "  - Precision: Proportion of predicted positive cases "
            "that are actually positive."
        )
        print("  - Recall: Proportion of actual positive cases correctly "
              "predicted.")
        print("  - F1-score: Harmonic mean of precision and recall, "
              "balancing both metrics.")
        print("  - Support: Number of actual occurrences of each class "
              "in the test set.")

        # Confusion Matrix:
        # True Positives, False Positives,
        # True Negatives, False Negatives
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        # Save confusion matrix plot
        cm_path = 'reports/confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path)
        plt.close()  # Close the plot to free up memory
        print(f"Confusion matrix plot saved to: {cm_path}")
        # Explanation of confusion matrix
        print("\nConfusion Matrix Interpretation:")
        print("  - Top-left (True Negatives): Correctly predicted "
              "non-diabetic patients.")
        print("  - Top-right (False Positives): Non-diabetic patients "
              "incorrectly predicted as diabetic.")
        print("  - Bottom-left (False Negatives): Diabetic patients")
        print("    incorrectly predicted as non-diabetic.")
        print("  - Bottom-right (True Positives): Correctly predicted "
              "diabetic patients.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure the model file exists by running model_training.py "
            "first."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")


def load_and_split_data(file_path):
    """
    Loads the diabetes dataset, splits it into features and target,
    and then further splits it into training and testing sets.

    Args:
        file_path (str): Path to the CSV file containing the diabetes dataset.

    Returns:
        X_train (pandas.DataFrame): The training set features.
        X_test (pandas.DataFrame): The test set features.
        y_train (pandas.Series): The training set target variable.
        y_test (pandas.Series): The test set target variable.
    Example:
        # Load and split data
        X_train, X_test, y_train, y_test = load_and_split_data(
            './data/diabetes.csv'
        )
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Separate features and target variable
        X = data.drop(columns=['Outcome'])
        y = data['Outcome']

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None, None, None


if __name__ == '__main__':
    """
    Main script entry point. Loads the diabetes dataset, splits it into
    training and testing sets,
    and then evaluates the model using the test set.

    Example:
        # Run the script
        python evaluate_model.py
    """
    # Load and split the dataset
    X_train, X_test, y_train, y_test = load_and_split_data(
        './data/diabetes.csv'
    )
    if X_test is not None and y_test is not None:
        # Evaluate the model
        evaluate_model(X_test, y_test)
    else:
        print("Data loading or splitting failed. Please check the dataset.")
