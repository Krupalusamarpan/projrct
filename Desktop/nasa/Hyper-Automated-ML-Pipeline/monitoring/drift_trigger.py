import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_PATH = "data/reference.csv"
CURRENT_PATH = "data/current.csv"

def detect_drift():
    try:
        reference = pd.read_csv(REFERENCE_PATH)
        current = pd.read_csv(CURRENT_PATH)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()
        drift_score = result['data_drift']['dataset_drift'] if 'data_drift' in result and result['data_drift'] else False
        print("Drift detected:", drift_score)
        if drift_score:
            print("🚨 Drift detected → Retraining model")
            os.system("python src/train.py")
        return drift_score
    except Exception as e:
        print("Drift check failed:", e)
        return False
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()
        drift_score = result['data_drift']['dataset_drift']
        print("Drift detected:", drift_score)
        if drift_score:
            print("🚨 Drift detected → Retraining model")
            os.system("python src/train.py")
    except Exception as e:
        print("Drift check failed:", e)

if __name__ == "__main__":
    detect_drift()
