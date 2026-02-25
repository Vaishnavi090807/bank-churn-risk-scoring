import os
import json
import joblib

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_pickle(obj, path: str) -> None:
    joblib.dump(obj, path)

def load_pickle(path: str):
    return joblib.load(path)