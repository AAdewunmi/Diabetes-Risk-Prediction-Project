import os
import subprocess

def run_script(script_name):
    """Runs a Python script using subprocess."""
    script_path = os.path.join('src', script_name)
    print(f"\n--- Running {script_name} ---")
    try:
        subprocess.run(['python', script_path], check=True)
        print(f"--- Finished running {script_name} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}")

if __name__ == "__main__":
    run_script('data_loading.py')
    run_script('data_processing.py')
    run_script('data_exploration.py')
    run_script('data_visualisation.py')
    run_script('statistical_analysis.py')
    run_script('model_training.py')
    run_script('model_evaluation.py')

    print("\n--- All scripts executed ---")