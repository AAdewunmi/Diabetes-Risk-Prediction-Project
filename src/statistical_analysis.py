import pandas as pd
from scipy import stats

def perform_statistical_analysis(df):
    """
    Performs basic statistical analysis on the diabetes dataset.

    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if df is None:
        return

    print("\n--- Statistical Analysis ---")

    # Example: Compare a numerical feature (e.g., 'Glucose') based on the 'Outcome'
    glucose_no_diabetes = df[df['Outcome'] == 0]['Glucose'].dropna()
    glucose_diabetes = df[df['Outcome'] == 1]['Glucose'].dropna()

    if not glucose_no_diabetes.empty and not glucose_diabetes.empty:
        t_statistic, p_value = stats.ttest_ind(glucose_no_diabetes, glucose_diabetes)
        print(f"\nT-test for Glucose vs. Outcome:")
        print(f"  T-statistic: {t_statistic:.3f}")
        print(f"  P-value: {p_value:.3f}")
        if p_value < 0.05:
            print("  Conclusion: There is a statistically significant difference in Glucose levels between those with and without diabetes.")
        else:
            print("  Conclusion: There is no statistically significant difference in Glucose levels based on this test.")
    else:
        print("\nNot enough data to perform t-test on Glucose vs. Outcome.")

    # Add more statistical tests as needed

if __name__ == '__main__':
    # Example usage
    try:
        analysis_data = pd.read_csv('../data/diabetes.csv') # Directly load
        perform_statistical_analysis(analysis_data)
    except FileNotFoundError:
        print("Please ensure 'diabetes.csv' is in the data directory.")