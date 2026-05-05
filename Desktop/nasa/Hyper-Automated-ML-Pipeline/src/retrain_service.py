import time
import os

MODEL_PATH = "models/model.pkl"

def monitor():
    while True:
        print("Checking system health...")
        if not os.path.exists(MODEL_PATH):
            print("❌ Model missing → Retraining")
            os.system("python src/train.py")
        # Trigger drift check
        os.system("python monitoring/drift_trigger.py")
        time.sleep(60)  # every 1 min

if __name__ == "__main__":
    monitor()

