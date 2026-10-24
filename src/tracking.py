from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "fraudshield-models"


def setup(experiment: str = EXPERIMENT):
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)


def log_run(run_name: str, params: dict, metrics: dict):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)