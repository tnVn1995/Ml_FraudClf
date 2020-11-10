from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data_path = file_dir.parent.parent / 'data'
    model_path = file_dir.parent.parent / 'models'
    random_state = 42
    
if __name__ == "__main__":
    print(file_dir)