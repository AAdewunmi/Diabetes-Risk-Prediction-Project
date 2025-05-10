import pandas as pd
from scipy import stats
import numpy as np

def perform_statistical_analysis(df):
    """
    Performs extended statistical tests on the diabetes dataset.

    Includes:
    - Independent t-test on 'Glucose' between Outcome groups.
    - Mann–Whitney U test on 'Insulin' between Outcome groups.
    - Pearson correlation between 'BMI' and 'Glucose'.
    - Chi-square test for independence between 'Pregnancies' (binned) and 'Outcome'.

    Args:
        df (pandas.DataFrame): DataFrame containing at least 'Outcome', 'Glucose', 'Insulin', 'BMI', and 'Pregnancies'.

    Returns:
        None

    Example:
        data = pd.read_csv('./data/diabetes.csv')
        perform_statistical_analysis(data)
    """
    if df is None:
        print("Error: DataFrame is None.")
        return

    print("\n--- Statistical Analysis ---")

    # --- T-Test: Glucose levels between outcomes ---
    glucose_0 = df[df['Outcome'] == 0]['Glucose'].dropna()
    glucose_1 = df[df['Outcome'] == 1]['Glucose'].dropna()
    if not glucose_0.empty and not glucose_1.empty:
        t_stat, p_val = stats.ttest_ind(glucose_0, glucose_1)
        print(f"\nT-Test on Glucose:\n  T-statistic = {t_stat:.3f}, P-value = {p_val:.3f}")
    else:
        print("\nNot enough data for t-test on Glucose.")

    # --- Mann–Whitney U Test: Insulin levels between outcomes ---
    insulin_0 = df[df['Outcome'] == 0]['Insulin'].dropna()
    insulin_1 = df[df['Outcome'] == 1]['Insulin'].dropna()
    if len(insulin_0) > 0 and len(insulin_1) > 0:
        u_stat, p_val = stats.mannwhitneyu(insulin_0, insulin_1, alternative='two-sided')
        print(f"\nMann–Whitney U Test on Insulin:\n  U-statistic = {u_stat:.3f}, P-value = {p_val:.3f}")
    else:
        print("\nNot enough data for Mann–Whitney U test on Insulin.")

    # --- Pearson Correlation: BMI and Glucose ---
    bmi = df['BMI'].dropna()
    glucose = df['Glucose'].dropna()
    if len(bmi) == len(glucose) and len(bmi) > 0:
        corr, p_val = stats.pearsonr(bmi, glucose)
        print(f"\nPearson Correlation (BMI vs. Glucose):\n  r = {corr:.3f}, P-value = {p_val:.3f}")
    else:
        print("\nBMI and Glucose length mismatch or insufficient data for correlation.")

    # --- Chi-Square Test: Binned Pregnancies vs Outcome ---
    # Bin pregnancies into categorical ranges
    df['Pregnancies_Binned'] = pd.cut(df['Pregnancies'], bins=[-1, 0, 2, 5, 10, np.inf],
                                      labels=['0', '1-2', '3-5', '6-10', '10+'])

    # Create contingency table
    contingency = pd.crosstab(df['Pregnancies_Binned'], df['Outcome'])
    if contingency.shape[0] > 1 and contingency.shape[1] == 2:
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square Test (Pregnancies_Binned vs. Outcome):\n  Chi2 = {chi2:.3f}, P-value = {p_val:.3f}, DoF = {dof}")
    else:
        print("\nInsufficient data to perform Chi-square test.")

if __name__ == '__main__':
    try:
        analysis_data = pd.read_csv('./data/diabetes.csv')
        perform_statistical_analysis(analysis_data)
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please ensure it is in the './data' directory.")
