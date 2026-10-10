"""Exporta una tabla ordenada de las corridas registradas en MLflow."""

from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DB = PROJECT_ROOT / "mlflow_experiments.db"
EXPERIMENT_NAME = "Bank_Marketing_Experimentacion"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "tables" / "experiment_comparison.csv"


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento {EXPERIMENT_NAME!r}.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    columns = [
        "tags.mlflow.runName",
        "params.C",
        "params.penalty",
        "params.class_weight",
        "params.tol",
        "params.seed",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
        "metrics.pr_auc",
        "run_id",
    ]
    available = [column for column in columns if column in runs.columns]
    comparison = runs[available].sort_values("metrics.pr_auc", ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(comparison.to_string(index=False))
    print(f"\nTabla exportada en: {OUTPUT_PATH}")
    if not comparison.empty:
        print(f"Mejor corrida por PR-AUC: {comparison.iloc[0]['tags.mlflow.runName']}")


if __name__ == "__main__":
    main()
