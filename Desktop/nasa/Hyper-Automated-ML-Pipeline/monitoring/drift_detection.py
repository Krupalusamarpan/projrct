from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_drift(reference,current):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference,current_data=current)
    report.save_html("drift_report.html")

