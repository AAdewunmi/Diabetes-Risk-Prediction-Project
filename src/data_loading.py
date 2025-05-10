import pandas as pd


def load_data(file_path):
    """
    Loads data from a CSV file into a Pandas DataFrame.

    This function attempts to read a CSV file from the given file path.
    If successful,
    it returns the loaded DataFrame.
    returns the loaded DataFrame. If the file is not found,
    an error message is printed.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame if successful,
        otherwise None.

    Raises:
        FileNotFoundError: If the file at the provided path does not exist.

    Example:
        # To load data from a file
        df = load_data('./data/diabetes.csv')
        if df is not None:
            print(df.head())  # Print the first few rows of the DataFrame
    """
    try:
        # Attempt to load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        # Handle the case where the file does not exist at the provided path
        print(f"Error: File not found at {file_path}")
        return None


if __name__ == '__main__':
    """
    The entry point of the script. It attempts to load a CSV file
    containing data
    from the specified file path, and if successful, prints the first few rows.

    Example:
        python load_data.py
    """
    # Define the path to the CSV file
    data_file_path = './data/diabetes.csv'

    # Load the data from the specified path
    data = load_data(data_file_path)

    # If data is successfully loaded, print the first 5 rows of the DataFrame
    if data is not None:
        print("\nFirst 5 rows of the loaded data:")
        print(data.head())  # Display the first 5 rows of the DataFrame
