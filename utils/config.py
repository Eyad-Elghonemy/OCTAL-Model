"""
config.py

Loads environment variables and the trained model artifacts once at
application startup, so they stay in memory for the lifetime of the
server instead of being reloaded on every request.
"""

from dotenv import load_dotenv
import os
import joblib


load_dotenv(override=True)

# App metadata, with safe fallbacks so a missing environment variable
# doesn't crash FastAPI's OpenAPI title/version assertion at import time.
APP_NAME = os.getenv("APP_NAME") or "Cars-Prediction"
VERSION = os.getenv("VERSION") or "1.0"

# Gates access to GET /logs only. There is no login/sign-up system in
# this project -- this single key is the only thing protecting the log.
SECRET_API_KEY = os.getenv("SECRET_API_KEY")

# Resolve paths relative to this file's location (not the current working
# directory), so the app runs correctly regardless of where uvicorn is
# launched from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER_PATH = os.path.join(BASE_DIR, "models")

PREPROCESSOR_PATH = os.path.join(MODELS_FOLDER_PATH, "preprocessor.pkl")
LINEAR_MODEL_PATH = os.path.join(MODELS_FOLDER_PATH, "lin_base.pkl")


def _load_model(path: str):
    """Loads a joblib artifact, raising a clear error if it's missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required model file not found: '{path}'.\n"
            f"Make sure a 'models/' folder exists at the project root "
            f"containing 'preprocessor.pkl' and 'lin_base.pkl' before "
            f"starting the application."
        )
    return joblib.load(path)


preprocessor = _load_model(PREPROCESSOR_PATH)
linear_model = _load_model(LINEAR_MODEL_PATH)
