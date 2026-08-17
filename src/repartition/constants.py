from pathlib import Path


COLOR2LABEL = {
    (255,   255,   255):   0,  # white -> not region of interest
    (200, 0,   0):   1,  # red → Tumor
    (0, 0, 200): 3,  # black → Necrosis
    (0, 200, 0): 2,  # green → Stroma
}

CLASSES = {
    'tumor': 1,
    'necrosis': 3,
    'stroma': 2,
}


MULTI_SLIDES_PATIENTS = (
    '24P41641',
    'DS_A02R',
    '25P31438',
    '26P2581',
    '25P35403',
    '25P32590',
    '24P43380',
    '25P22227'
)

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
