import os
import subprocess


def run_script(script_name):
    """
    Runs a Python script located in the 'src' directory using subprocess,
    and saves the output to a log file.

    Args:
        script_name (str): The name of the Python script file to execute.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the subprocess execution fails.
        FileNotFoundError: If the specified script file does not exist.

    Example:
        run_script('data_loading.py')
    """
    # Ensure the reports directory exists
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Construct the full path to the script
    script_path = os.path.join('src', script_name)

    # Define the log file path
    log_file = os.path.join(
        'reports', f"{script_name.replace('.py', '')}_log.txt"
    )

    # Open log file to capture output
    with open(log_file, 'w') as log:
        print(f"\n--- Running {script_name} ---", file=log)
        try:
            # Execute the script using subprocess
            # and redirect output to the log file
            subprocess.run(
                ['python', script_path], check=True, stdout=log, stderr=log
            )
            print(f"--- Finished running {script_name} ---", file=log)
        except subprocess.CalledProcessError as e:
            # Handle errors during script execution
            print(f"Error running {script_name}: {e}", file=log)
        except FileNotFoundError:
            # Handle case where the script is not found
            print(f"Error: Script not found at {script_path}", file=log)


def main():
    """
    Executes a predefined list of Python scripts in sequence
    and saves the output
    to log files in the 'reports' folder.
    """
    scripts_to_run = [
        'data_loading.py',
        'data_processing.py',
        'data_exploration.py',
        'data_visualisation.py',
        'statistical_analysis.py',
        'model_training.py',
        'model_evaluation.py'
        'model_explainability_interpretability.py'
    ]

    for script in scripts_to_run:
        run_script(script)

    print("\n--- All scripts executed. Logs are saved in the "
          "'reports' folder.")


if __name__ == "__main__":
    # Entry point of the script
    main()
