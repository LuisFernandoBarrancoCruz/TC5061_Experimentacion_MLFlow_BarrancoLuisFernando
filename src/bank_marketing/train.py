"""Entrenamiento parametrizado de Bank Marketing con MLflow Tracking.

Ejemplo:
    python src/bank_marketing/train.py --C 1.0 --penalty l2 --class-weight balanced --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


UCI_PACKAGE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


def parse_args() -> argparse.Namespace:
    """Define y valida los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena una regresión logística sobre UCI Bank Marketing y registra "
            "parámetros, métricas, artefactos y modelo en MLflow."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/bank-full.csv"),
        help="Ruta al archivo bank-full.csv.",
    )
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow_experiments.db"),
        help="Base SQLite usada como backend local de MLflow.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Bank_Marketing_Experimentacion",
        help="Nombre que agrupará todas las corridas en MLflow.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre opcional de la corrida; se genera automáticamente si se omite.",
    )
    parser.add_argument(
        "--C",
        dest="c_value",
        type=float,
        default=1.0,
        help="Inverso de la regularización; valores menores regularizan más.",
    )
    parser.add_argument(
        "--penalty",
        choices=("l1", "l2"),
        default="l2",
        help="Tipo de penalización de la regresión logística.",
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Ponderación de clases; 'none' desactiva el balance automático.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerancia del criterio de convergencia.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones del optimizador.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral aplicado a la probabilidad para producir la clase final.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporción reservada para prueba.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla usada en NumPy, Python, split y modelo.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="No descargar los datos cuando --data-path no exista.",
    )
    args = parser.parse_args()

    if args.c_value <= 0:
        parser.error("--C debe ser mayor que cero.")
    if args.tol <= 0:
        parser.error("--tol debe ser mayor que cero.")
    if args.max_iter <= 0:
        parser.error("--max-iter debe ser mayor que cero.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1.")
    if not 0 < args.threshold < 1:
        parser.error("--threshold debe estar entre 0 y 1.")
    return args


def set_global_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad controlables por el script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def download_dataset(csv_path: Path) -> None:
    """Descarga y extrae el archivo oficial de UCI cuando no existe localmente."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    outer_zip = csv_path.parent / "bank_marketing.zip"
    inner_zip = csv_path.parent / "bank.zip"

    try:
        print("Descargando el paquete oficial de UCI...")
        urlretrieve(UCI_PACKAGE_URL, outer_zip)
        with ZipFile(outer_zip) as archive:
            archive.extract("bank.zip", csv_path.parent)
    except Exception as exc:
        print(f"Ruta principal no disponible ({exc}). Usando respaldo...")
        urlretrieve(UCI_FALLBACK_URL, inner_zip)

    if not inner_zip.exists():
        raise FileNotFoundError("No se encontró bank.zip después de la descarga.")

    with ZipFile(inner_zip) as archive:
        archive.extract("bank-full.csv", csv_path.parent)


def load_data(csv_path: Path, allow_download: bool = True) -> pd.DataFrame:
    """Carga el dataset y verifica las columnas mínimas esperadas."""
    if not csv_path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"No existe {csv_path}. Retira --no-download o copia bank-full.csv."
            )
        download_dataset(csv_path)

    data = pd.read_csv(csv_path, sep=";")
    required_columns = {"y", "duration", "pdays"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"El dataset no contiene las columnas requeridas: {sorted(missing)}")
    return data


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Elimina duplicados y crea el objetivo y una feature reproducible."""
    cleaned = data.drop_duplicates().reset_index(drop=True).copy()
    cleaned["target"] = cleaned["y"].map({"no": 0, "yes": 1})
    if cleaned["target"].isna().any():
        raise ValueError("La variable y contiene valores distintos de 'yes' y 'no'.")
    cleaned["target"] = cleaned["target"].astype(int)
    cleaned["previously_contacted"] = (cleaned["pdays"] != -1).astype(int)

    summary = {
        "rows_raw": int(len(data)),
        "rows_clean": int(len(cleaned)),
        "duplicates_removed": int(len(data) - len(cleaned)),
        "columns_raw": int(data.shape[1]),
        "positive_rate": float(cleaned["target"].mean()),
    }
    return cleaned, summary


def split_data(
    data: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa train/test de forma estratificada y excluye fuga operacional."""
    # duration sólo se conoce después de la llamada; no debe usarse para decidir a quién llamar.
    features = data.drop(columns=["y", "target", "duration"])
    target = data["target"]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )


def build_pipeline(
    features: pd.DataFrame,
    c_value: float,
    penalty: str,
    class_weight: str,
    tol: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    """Construye preprocesamiento y modelo dentro de un solo Pipeline."""
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    estimator = LogisticRegression(
        C=c_value,
        penalty=penalty,
        class_weight=None if class_weight == "none" else class_weight,
        solver="liblinear",
        tol=tol,
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calcula métricas coherentes con un problema binario desbalanceado."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
    }
    return metrics, predictions, probabilities


def save_evaluation_artifacts(
    artifact_dir: Path,
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, float],
    parameters: dict[str, object],
) -> None:
    """Genera archivos que facilitan la auditoría de la corrida."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(target, predictions, digits=4, zero_division=0)
    (artifact_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifact_dir / "run_parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    matrix = confusion_matrix(target, predictions)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(matrix, display_labels=["no", "yes"]).plot(
        cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(artifact_dir / "confusion_matrix.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(target, probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(target, probabilities)
    prevalence = float(target.mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(fpr, tpr, color="#2563EB", label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Curva ROC", xlabel="Falsos positivos", ylabel="Recall")
    axes[0].legend()

    axes[1].plot(
        recall_curve,
        precision_curve,
        color="#0F766E",
        label=f"AP = {metrics['pr_auc']:.3f}",
    )
    axes[1].axhline(prevalence, linestyle="--", color="gray", label="Prevalencia")
    axes[1].set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precisión")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "roc_pr_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def get_git_commit() -> str:
    """Devuelve el commit actual cuando el proyecto está dentro de Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "not_available"


def configure_mlflow(tracking_db: Path, experiment_name: str) -> str:
    """Configura un backend SQLite absoluto y un experimento nombrado."""
    resolved_db = tracking_db.resolve()
    resolved_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{resolved_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def make_run_name(args: argparse.Namespace) -> str:
    """Genera un nombre corto que deja visible la configuración principal."""
    weight = "bal" if args.class_weight == "balanced" else "none"
    return (
        f"lr_C{args.c_value:g}_{args.penalty}_cw-{weight}_"
        f"tol{args.tol:g}_seed{args.seed}"
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    raw_data = load_data(args.data_path, allow_download=not args.no_download)
    clean_data_frame, data_summary = clean_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(
        clean_data_frame, test_size=args.test_size, seed=args.seed
    )

    model = build_pipeline(
        X_train,
        c_value=args.c_value,
        penalty=args.penalty,
        class_weight=args.class_weight,
        tol=args.tol,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    model.fit(X_train, y_train)
    metrics, predictions, probabilities = evaluate_model(
        model, X_test, y_test, threshold=args.threshold
    )

    tracking_uri = configure_mlflow(args.tracking_db, args.experiment_name)
    run_name = args.run_name or make_run_name(args)
    parameters = {
        "model": "LogisticRegression",
        "C": args.c_value,
        "penalty": args.penalty,
        "class_weight": args.class_weight,
        "solver": "liblinear",
        "tol": args.tol,
        "max_iter": args.max_iter,
        "threshold": args.threshold,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "duration_excluded": True,
        "numeric_scaler": "RobustScaler",
        "categorical_encoder": "OneHotEncoder",
    }

    with tempfile.TemporaryDirectory(prefix="bank_marketing_artifacts_") as temp_dir:
        artifact_dir = Path(temp_dir)
        save_evaluation_artifacts(
            artifact_dir,
            y_test,
            predictions,
            probabilities,
            metrics,
            parameters,
        )
        (artifact_dir / "data_summary.json").write_text(
            json.dumps(data_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "project": "TC5061 Bank Marketing",
                    "task": "binary_classification",
                    "dataset": "UCI Bank Marketing",
                    "git_commit": get_git_commit(),
                    "reproducible_seed": str(args.seed),
                }
            )
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")

            input_example = X_train.head(5).copy()
            signature = infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="pickle",
                signature=signature,
                input_example=input_example,
            )
            run_id = run.info.run_id

    print("\nCorrida registrada correctamente")
    print(f"Run name: {run_name}")
    print(f"Run ID:   {run_id}")
    print(f"Tracking: {tracking_uri}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:>10}: {value:.6f}")


if __name__ == "__main__":
    main()
