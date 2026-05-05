'''Retraining trigger - comprehensive monitoring'''
import mlflow
import pandas as pd
from datetime import datetime, timedelta
from monitoring.drift_trigger import detect_drift
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Thresholds
MAE_THRESHOLD = 10.0
DRIFT_THRESHOLD = 0.5
DAYS_SINCE_TRAIN = 7

def should_retrain():
    """Comprehensive retrain decision logic"""
    issues = []
    
    # 1. Metric degradation
    try:
        exp = mlflow.get_experiment_by_name("RUL_prediction")
        if exp:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
            latest_mae = runs.iloc[0]['metrics.mae'] if not runs.empty else float('inf')
        else:
            latest_mae = float('inf')
    except:
        latest_mae = float('inf')
    if latest_mae > MAE_THRESHOLD:
        issues.append(f"MAE degradation: {latest_mae:.2f} > {MAE_THRESHOLD}")
    
    # 2. Data drift
    try:
        drift_detected = detect_drift()
        if drift_detected:
            issues.append("Data drift detected")
    except:
        issues.append("Drift check failed")
    
    # 3. Time since last training
    try:
        exp = mlflow.get_experiment_by_name("RUL_prediction")
        if exp:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
            if not runs.empty:
                last_run = runs.iloc[0]
                last_train = datetime.fromtimestamp(last_run['start_time']/1000)
                if datetime.now() - last_train > timedelta(days=DAYS_SINCE_TRAIN):
                    issues.append("Past retrain schedule")
    except:
        pass
    
    retrain = len(issues) > 0
    print(f"Retrain decision: {retrain}")
    if issues:
        print("Reasons:", issues)
    
    return retrain

if __name__ == "__main__":
    if should_retrain():
        print("🚀 TRIGGERING RETRAIN")
        os.system("python src/train.py")
    else:
        print("✅ Model OK - no retrain needed")

