import joblib
import os

MODEL_PATH = "models/model.pkl"
BACKUP_PATH = "models/backup_model.pkl"

def save_model(model):
    if os.path.exists(MODEL_PATH):
        os.replace(MODEL_PATH, BACKUP_PATH)
    joblib.dump(model, MODEL_PATH)

def rollback():
    if os.path.exists(BACKUP_PATH):
        os.replace(BACKUP_PATH, MODEL_PATH)
        print("⚠️ Rolled back to previous model")

