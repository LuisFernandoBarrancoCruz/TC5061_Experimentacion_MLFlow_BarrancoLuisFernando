# Experimentación base con MLflow — UCI Bank Marketing

Proyecto individual de **TC5061 MLOps**. Convierte el baseline de la semana anterior en un script parametrizado y reproducible que registra cada corrida en MLflow Tracking.

## Autor

- Luis Fernando Barranco Cruz
- Matrícula: A01797868

## Objetivo

Predecir si un cliente contratará un depósito a plazo utilizando el dataset **UCI Bank Marketing**. El modelo es una regresión logística integrada en un `Pipeline` de Scikit-learn.

La variable `duration` se excluye porque sólo se conoce después de la llamada y produciría fuga de información si el modelo se utilizara para decidir a quién contactar.

## Estructura

```text
.
├── notebooks/            # notebook anterior, reservado para EDA
├── src/bank_marketing/
│   └── train.py          # carga, limpieza, entrenamiento, evaluación y MLflow
├── scripts/
│   ├── run_experiments.py
│   └── summarize_runs.py
├── data/raw/             # datos originales; no se versionan
├── reports/
│   ├── figures/          # capturas y figuras para la entrega
│   └── tables/           # comparación exportada
├── tests/                # espacio para pruebas de evolución
├── requirements.txt      # dependencias con versiones fijas
├── .gitignore
└── README.md
```

La estructura está inspirada en **Cookiecutter Data Science**. Los datos, los artefactos de MLflow y la base `mlflow_experiments.db` se generan localmente y no se versionan.

## Instalación

Probado con Python 3.13 y MLflow 3.16.1.

```bash
python3 -m venv venv3.13
source venv3.13/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución con valores por defecto

Si `data/raw/bank-full.csv` no existe, el script intenta descargarlo desde UCI.

```bash
python src/bank_marketing/train.py
```

Ejemplo parametrizado:

```bash
python src/bank_marketing/train.py \
  --C 1.0 \
  --penalty l2 \
  --class-weight balanced \
  --tol 0.0001 \
  --max-iter 1000 \
  --seed 42
```

Ayuda completa:

```bash
python src/bank_marketing/train.py --help
```

## Seis corridas comparables

El siguiente comando ejecuta seis configuraciones con la misma semilla, el mismo split y el mismo experimento:

```bash
python scripts/run_experiments.py
```

| Corrida | C | Penalización | Class weight | Tolerancia |
|---|---:|---|---|---:|
| 01 | 0.01 | l2 | balanced | 0.0001 |
| 02 | 0.1 | l2 | balanced | 0.0001 |
| 03 | 1.0 | l2 | balanced | 0.0001 |
| 04 | 10.0 | l2 | balanced | 0.0001 |
| 05 | 1.0 | l1 | balanced | 0.0001 |
| 06 | 1.0 | l2 | none | 0.0001 |

Después se puede generar una tabla local ordenada por PR-AUC:

```bash
python scripts/summarize_runs.py
```

## Abrir MLflow UI

En una terminal separada y desde esta carpeta:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow_experiments.db \
  --host 127.0.0.1 \
  --port 5001
```

Abrir: <http://127.0.0.1:5001>

Se usa el puerto `5001` para evitar conflicto con instancias anteriores en `5000`. Si está libre, también puede utilizarse `--port 5000`.

## Información registrada en cada corrida

- **Parámetros:** `C`, `penalty`, `class_weight`, `tol`, `max_iter`, `threshold`, `seed`, `test_size` y configuración del preprocesamiento.
- **Métricas:** accuracy, precision, recall, F1, ROC-AUC y PR-AUC.
- **Artefactos:** matriz de confusión, curvas ROC/PR, reporte de clasificación, métricas, parámetros y resumen de datos.
- **Modelo:** Pipeline completo con preprocesamiento y regresión logística, firma e input example.
- **Trazabilidad:** nombre del proyecto, dataset, tarea, semilla y commit de Git cuando está disponible.

## Reproducibilidad

La semilla `42` se fija en Python, NumPy, `train_test_split` y `LogisticRegression`. Las seis corridas utilizan exactamente el mismo split estratificado. Las dependencias están fijadas en `requirements.txt`.

Para comprobar la repetibilidad, ejecuta dos veces el mismo comando y verifica que las métricas coincidan:

```bash
python src/bank_marketing/train.py --run-name reproducibility_A --C 1 --penalty l2 --class-weight balanced --seed 42
python src/bank_marketing/train.py --run-name reproducibility_B --C 1 --penalty l2 --class-weight balanced --seed 42
```

## Evidencias para el PDF

1. En la vista **Runs**, selecciona las seis corridas y abre **Compare**. Muestra `C`, `penalty`, `class_weight`, F1, ROC-AUC y PR-AUC.
2. Captura la tabla comparativa con los nombres y valores legibles.
3. Abre la mejor corrida y captura parámetros, métricas y estado `Finished`.
4. En la pestaña **Artifacts**, muestra `evaluation/` y el modelo registrado.

## Fuente de datos

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>
