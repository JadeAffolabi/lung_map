from pathlib import Path


COLOR2LABEL = {
    (255,   255,   255):   0,  # white -> not region of interest
    (200, 0,   0):   1,  # red (#c80000) → Tumor
    (150, 200, 150): 2,  # green (#96c896) → Stroma
    (50, 50, 50): 3,  # black (#323232) → Necrosis
}

CLASSES = {
    'tumor': 1,
    'stroma': 2,
    'necrosis': 3
}

CLASS_NAMES = ["necrosis", "stroma", "tumor", "tumor_bed"]

def find_project_root() -> Path:
    # Use __file__ if running as a script, fallback to cwd() if in a notebook
    current_path = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    for path in [current_path] + list(current_path.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    raise FileNotFoundError("Could not find project root (missing pyproject.toml or .git)")

PROJECT_ROOT = find_project_root()
PATH_PATIENT_DATA = PROJECT_ROOT / 'data' / 'repartition' / 'donnees_patients.xlsx'
