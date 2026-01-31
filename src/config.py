from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print('BASE_DIR:', BASE_DIR)
