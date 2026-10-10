"""Ejecuta seis configuraciones comparables usando el mismo train.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "bank_marketing" / "train.py"


EXPERIMENTS = [
    {
        "run_name": "01_C001_l2_balanced",
        "C": "0.01",
        "penalty": "l2",
        "class_weight": "balanced",
        "tol": "0.0001",
    },
    {
        "run_name": "02_C01_l2_balanced",
        "C": "0.1",
        "penalty": "l2",
        "class_weight": "balanced",
        "tol": "0.0001",
    },
    {
        "run_name": "03_C1_l2_balanced",
        "C": "1.0",
        "penalty": "l2",
        "class_weight": "balanced",
        "tol": "0.0001",
    },
    {
        "run_name": "04_C10_l2_balanced",
        "C": "10.0",
        "penalty": "l2",
        "class_weight": "balanced",
        "tol": "0.0001",
    },
    {
        "run_name": "05_C1_l1_balanced",
        "C": "1.0",
        "penalty": "l1",
        "class_weight": "balanced",
        "tol": "0.0001",
    },
    {
        "run_name": "06_C1_l2_no_weight",
        "C": "1.0",
        "penalty": "l2",
        "class_weight": "none",
        "tol": "0.0001",
    },
]


def main() -> None:
    for index, config in enumerate(EXPERIMENTS, start=1):
        print(f"\n[{index}/{len(EXPERIMENTS)}] Ejecutando {config['run_name']}")
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--run-name",
            config["run_name"],
            "--C",
            config["C"],
            "--penalty",
            config["penalty"],
            "--class-weight",
            config["class_weight"],
            "--tol",
            config["tol"],
            "--seed",
            "42",
            "--experiment-name",
            "Bank_Marketing_Experimentacion",
            "--tracking-db",
            "mlflow_experiments.db",
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    print("\nTodas las corridas terminaron correctamente.")
    print(
        "Abre la interfaz con: mlflow ui --backend-store-uri "
        "sqlite:///mlflow_experiments.db --host 127.0.0.1 --port 5001"
    )


if __name__ == "__main__":
    main()
