from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from train import train_model

dag = DAG(
    "ml_training_pipeline",
    start_date=datetime(2024,1,1),
    schedule_interval="@daily"
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

