import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os


def train_model(df):
    """
    Trains a logistic regression model to predict diabetes.
    Saves the trained model.

    This function:
    1. Splits the input dataset into features (X) and target (y).
    2. Scales the numerical features using StandardScaler.
    3. Trains a logistic regression model on the preprocessed data.
    4. Saves the trained model in the 'models' directory.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the diabetes
        dataset.
        It should include the features and the target variable ('Outcome').

    Returns:
        model (sklearn.pipeline.Pipeline): The trained model pipeline
        (preprocessing + classifier).
        X_test (pandas.DataFrame): The test set features.
        y_test (pandas.Series): The test set target variable.

    Example:
        # Load data
        data = pd.read_csv('./data/diabetes.csv')
        # Train the model
        model, X_test, y_test = train_model(data)
    """
    if df is None:
        print("Error: DataFrame is empty or None.")
        return None

    # Define features (X) and target (y)
    X = df.drop(columns=['Outcome'])  # Features: All columns except 'Outcome'
    y = df['Outcome']  # Target variable: 'Outcome' column

    # Create preprocessing pipeline for scaling numerical features
    # Scaling the data to standardize the features
    numerical_transformer = StandardScaler()
    preprocessor = Pipeline(steps=[('scaler', numerical_transformer)])

    # Create the full pipeline combining preprocessing
    # and the logistic regression classifier
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(
                                solver='liblinear', random_state=42))])

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into training and testing sets.")

    # Train the model on the training set
    model.fit(X_train, y_train)
    print("Logistic Regression model trained.")

    # Create the 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the trained model as a .joblib file for later use
    model_path = 'models/diabetes_prediction_model.joblib'
    joblib.dump(model, model_path)
    print(f"Trained model saved as {model_path}")

    # Return the trained model and the test set for evaluation
    return model, X_test, y_test


if __name__ == '__main__':
    """
    Main entry point of the script. Loads the diabetes dataset, trains a
    logistic regression model,
    and saves the trained model to a file.

    Example:
        python train_model.py
    """
    try:
        # Load the diabetes dataset
        training_data = pd.read_csv('./data/diabetes.csv')
        # Train the model and save it
        train_model(training_data)
    except FileNotFoundError:
        print("Error: 'diabetes.csv' file not found in the data directory.")
