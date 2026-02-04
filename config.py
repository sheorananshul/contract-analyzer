# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Explicitly load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Make sure .env exists in project root and contains OPENAI_API_KEY"
    )
