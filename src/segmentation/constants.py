from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

CLASSES = ['other', 'tumor', 'necrosis', 'stroma']
CLASSES_TO_LABELS = {
    'other': 0,
    'tumor': 1,
    'necrosis': 2,
    'stroma': 3
}

LABELS_TO_CLASSES = {
    0: 'other',
    1: 'tumor',
    2: 'necrosis',
    3: 'stroma',
}

CLASS_CONVERSION_LUAD = {
    'tumor': [0],
    'necrosis': [1],
    'stroma': [2, 3],
}
CLASS_CONVERSION_BCSS = {
    'tumor': [1, 19, 20],
    'necrosis': [4],
    'stroma': [2, 3, 10, 11, 14],
}

def find_project_root() -> Path:
    # Use __file__ if running as a script, fallback to cwd() if in a notebook
    current_path = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    for path in [current_path] + list(current_path.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    raise FileNotFoundError("Could not find project root (missing pyproject.toml or .git)")

PROJECT_ROOT = find_project_root()
PATH_SEG_DATA = PROJECT_ROOT / 'data' / 'segmentation'
RAW_DATA_DIR =  str(PATH_SEG_DATA / 'raw_data')
DIR_LUAD = 'LUAD-HistoSeg'
DIR_BCSS = ""

ACCESS_TOKEN = os.getenv('TOKEN_HUGGINGFACE')
