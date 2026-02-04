import json
from pathlib import Path


def load_standards(path: str = "standards/compliance_standards.json") -> dict:
    standards_path = Path(path)
    if not standards_path.exists():
        raise FileNotFoundError(f"Standards file not found: {standards_path.resolve()}")

    with standards_path.open("r", encoding="utf-8") as f:
        return json.load(f)
