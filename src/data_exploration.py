import pandas as pd

def explore_data(df):
    """
    Performs exploratory data analysis on the diabetes dataset.

    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if df is None:
        return

    print("\n--- Basic Information ---")
    df.info()

    print("\n--- Summary Statistics ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Add more exploration as needed based on the 'diabetes.csv' dataset

if __name__ == '__main__':
    # Example usage
    try:
        explored_data = pd.read_csv('../data/diabetes.csv') # Directly load for independent run
        explore_data(explored_data)
    except FileNotFoundError:
        print("Please ensure 'diabetes.csv' is in the data directory.")