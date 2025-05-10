import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    """
    Creates visualizations for the diabetes dataset.

    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if df is None:
        return

    print("\n--- Data Visualisation ---")

    # Example: Distribution of the target variable ('Outcome')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df)
    plt.title('Distribution of Diabetes Outcome')
    plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
    plt.ylabel('Count')
    plt.savefig('../reports/outcome_distribution.png')
    plt.close()
    print("Outcome distribution plot saved to reports/outcome_distribution.png")

    # Add more visualizations relevant to the 'diabetes.csv' dataset
    # For example, histograms of numerical features, box plots, etc.

if __name__ == '__main__':
    # Example usage
    try:
        visualisation_data = pd.read_csv('../data/diabetes.csv') # Directly load
        visualize_data(visualisation_data)
    except FileNotFoundError:
        print("Please ensure 'diabetes.csv' is in the data directory.")