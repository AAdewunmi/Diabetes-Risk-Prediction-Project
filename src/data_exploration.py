import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """
    Performs exploratory data analysis (EDA) on a diabetes dataset.

    This function provides a detailed overview of the dataset, including:
    - Basic information (data types, non-null counts)
    - Summary statistics
    - Missing values
    - Exploratory visualizations such as scatter plots, correlation heatmaps,
      and box plots.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the diabetes
            dataset.
    Returns:
        None

    Example:
        df = pd.read_csv('path_to_diabetes_data.csv')
        explore_data(df)
    """
    if df is None:
        print("No data provided.")
        return

    # Display basic information about the dataset
    print("\n--- Basic Information ---")
    df.info()  # Display data types and non-null counts

    # Display summary statistics of the dataset
    print("\n--- Summary Statistics ---")
    print(df.describe())  # Display statistical summary for numerical columns

    # Check for missing values in the dataset
    print("\n--- Missing Values ---")
    # Show the count of missing values for each column
    print(df.isnull().sum())

    # Exploratory visualizations and insights
    print("\n--- Exploratory Visualizations and Insights ---")

    # 1. Glucose vs. Age scatter plot, colored by Outcome (Diabetes diagnosis)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df)
    plt.title('Glucose Level vs. Age by Diabetes Outcome')
    plt.xlabel('Age')
    plt.ylabel('Glucose Level')
    plt.savefig('reports/eda_glucose_vs_age_outcome.png')
    # Close the plot to avoid display in Jupyter notebooks or
    # interactive environments
    plt.close()
    print(
        "EDA: Glucose vs. Age by Outcome plot saved to "
        "/reports/eda_glucose_vs_age_outcome.png"
    )

    # 2. Correlation heatmap of BMI, Glucose, and BloodPressure
    correlation_cols = ['BMI', 'Glucose', 'BloodPressure']
    # Calculate correlation matrix for specified columns
    corr_matrix = df[correlation_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, cmap='coolwarm', fmt=".2f"
    )  # Heatmap with annotation
    plt.title('Correlation Heatmap of BMI, Glucose, BloodPressure')
    plt.savefig('reports/eda_correlation_bmi_glucose_bp.png')
    plt.close()  # Close the plot
    print(
        "EDA: Correlation heatmap of BMI, Glucose, BloodPressure saved to "
        "/reports/eda_correlation_bmi_glucose_bp.png"
    )

    # 3. Box plot of Insulin levels by Outcome (0: No Diabetes, 1: Diabetes)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Outcome', y='Insulin', data=df)
    plt.title('Insulin Distribution by Diabetes Outcome')
    plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
    plt.ylabel('Insulin Level')
    plt.savefig('reports/eda_insulin_by_outcome_boxplot.png')
    plt.close()  # Close the plot
    print(
        "EDA: Insulin distribution by Outcome (boxplot) saved to "
        "/reports/eda_insulin_by_outcome_boxplot.png"
    )


if __name__ == '__main__':
    """
    The entry point of the script. It attempts to load the diabetes dataset
    from a CSV file located in the './data' directory. If successful, it
    calls the explore_data() function to perform exploratory data analysis.
    If the dataset file is not found, it prints an error message.
    This script is designed to be run as a standalone program.
    It loads the diabetes dataset into a pandas DataFrame from a CSV file,
    and then performs exploratory data analysis using the
    explore_data() function.

    Usage:
        Run this script after ensuring the 'diabetes.csv' file
        is located in the './data' directory.
    """
    try:
        # Load the diabetes dataset into a pandas DataFrame
        exploration_data = pd.read_csv('./data/diabetes.csv')

        # Perform exploratory data analysis (EDA) on the loaded dataset
        explore_data(exploration_data)

    except FileNotFoundError:
        # Handle the case when the dataset file is not found
        print("Please ensure 'diabetes.csv' is in the data directory.")
