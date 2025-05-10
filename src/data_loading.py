import pandas as pd


def load_data(file_path):
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

if __name__ == '__main__':
    # This block will only run if this script is executed directly
    data = load_data('../data/diabetes.csv')
    if data is not None:
        print(data.head())