import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data(df):
    """
    Creates visualizations for the diabetes dataset.

    This function generates various visualizations to explore and analyze
    the diabetes dataset.
    The plots are saved in the 'reports' directory for further analysis
    or reporting.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the diabetes
            dataset.

    Returns:
        None

    Example:
        # Assuming df is the DataFrame loaded from a CSV file
        df = pd.read_csv('./data/diabetes.csv')
        visualize_data(df)
    """
    # Check if the input DataFrame is valid
    if df is None:
        print("Error: The input DataFrame is None.")
        return

    print("\n--- Data Visualisation ---")

    # 1. Distribution of the target variable ('Outcome')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df)
    plt.title('Distribution of Diabetes Outcome')
    plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
    plt.ylabel('Count')
    plt.savefig('reports/outcome_distribution.png')
    plt.close()  # Close the plot to avoid display issues when saving
    print("Outcome distribution plot saved to "
          "/reports/outcome_distribution.png")

    # 2. Pairplot of selected features colored by Outcome
    selected_features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'Age', 'Outcome'
    ]
    sns.pairplot(df[selected_features], hue='Outcome', diag_kind='kde')
    plt.savefig('reports/pairplot_selected_features.png')
    plt.close()  # Close the plot
    print("Pairplot of selected features saved to "
          "/reports/pairplot_selected_features.png")

    # 3. Violin plot of BMI by Outcome
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Outcome', y='BMI', data=df)
    plt.title('BMI Distribution by Diabetes Outcome')
    plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
    plt.ylabel('BMI')
    plt.savefig('reports/bmi_distribution_by_outcome.png')
    plt.close()  # Close the plot
    print("BMI distribution by Outcome (violin plot) saved to "
          "/reports/bmi_distribution_by_outcome.png")

    # 4. Heatmap of correlations between features
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()  # Calculate correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Features')
    plt.savefig('reports/correlation_heatmap.png')
    plt.close()  # Close the plot
    print("Correlation heatmap saved to /reports/correlation_heatmap.png")


if __name__ == '__main__':
    """
    The entry point of the script. It loads the diabetes dataset, performs
    visualizations, and saves the resulting plots in the 'reports' directory.

    Example:
        python visualize_data.py
    """
    try:
        # Load the diabetes dataset from the CSV file
        visualisation_data = pd.read_csv('./data/diabetes.csv')
        # Perform the data visualizations
        visualize_data(visualisation_data)

    except FileNotFoundError:
        print(
            "Error: 'diabetes.csv' file not found. Please ensure it is in the "
            "'./data' directory."
        )
