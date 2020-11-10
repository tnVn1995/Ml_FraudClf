from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data_path = file_dir.parent / 'data'
    model_path = file_dir.parent / 'models'
    random_state = 42
    reports = file_dir.parent / 'reports'

if __name__ == "__main__":
    print(file_dir)
    print(CONFIG.reports)