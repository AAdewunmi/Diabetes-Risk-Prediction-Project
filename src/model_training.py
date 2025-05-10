import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_model(df):
    """
    Trains a logistic regression model to predict diabetes.

    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if df is None:
        return None

    # Define features (X) and target (y)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Identify numerical features for scaling

    # Create preprocessing pipeline for numerical features
    numerical_transformer = StandardScaler()
    preprocessor = Pipeline(steps=[('scaler', numerical_transformer)])

    # Create the full pipeline with preprocessing and the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model.fit(X_train, y_train)
    print("Logistic Regression model trained.")

    # Save the trained model
    joblib.dump(model, '../models/diabetes_prediction_model.joblib')
    print("Trained model saved as ../models/diabetes_prediction_model.joblib")

    return model, X_test, y_test

if __name__ == '__main__':
    # Example usage
    try:
        training_data = pd.read_csv('../data/diabetes.csv') # Directly load
        train_model(training_data)
    except FileNotFoundError:
        print("Please ensure 'diabetes.csv' is in the data directory.")