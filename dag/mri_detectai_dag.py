from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_googlenet():
    from src.pretraitement import googleeNet

    googleeNet()


def run_resnet():
    from src.pretraitement import resNetTrin

    resNetTrin()


def run_yolo():
    from src.yolo import train_yolo

    train_yolo()


def run_yolo_evaluation():
    from src.evaluationYolo import evaluate_yolo

    evaluate_yolo()


def run_evaluation():
    from src.evaluation import evaluate

    evaluate()

with DAG(
    dag_id="mri_detectai_pipeline",
    description="Automatisation du pipeline ML pour MRI DetectAI",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args={
        "owner": "mri_detectai",
        "retries": 1,
    },
    tags=["deep-learning", "MRI", "tensorflow"],
) as dag:

    googleeNet = PythonOperator(
        task_id="googleeNet",
        python_callable=run_googlenet,
    )

    resNetTrin = PythonOperator(
        task_id="resNetTrin",
        python_callable=run_resnet,
    )

    yolo = PythonOperator(
        task_id="yolo",
        python_callable=run_yolo,
    )

    evaluationYolo = PythonOperator(
        task_id="evaluationYolo",
        python_callable=run_yolo_evaluation,
    )

    evaluation = PythonOperator(
        task_id="evaluation",
        python_callable=run_evaluation,
    )

    googleeNet >> resNetTrin >> yolo >> evaluationYolo >> evaluation