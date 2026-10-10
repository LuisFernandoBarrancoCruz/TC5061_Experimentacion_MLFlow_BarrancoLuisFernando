# Checklist contra la rúbrica

## 1. Migración del notebook a script — 20%

- [x] El notebook se reserva para EDA.
- [x] `src/bank_marketing/train.py` está dividido en funciones con responsabilidades claras.
- [x] No se duplica la lógica de preprocesamiento entre entrenamiento e inferencia.

## 2. Parametrización — 15%

- [x] `C`, `penalty`, `class_weight`, `tol` y `max_iter` se exponen por CLI.
- [x] La semilla se expone con `--seed`.
- [x] Todos los argumentos tienen valores por defecto y ayuda descriptiva.

## 3. MLflow Tracking — 25%

- [x] Parámetros y seis métricas registrados.
- [x] Matriz de confusión, curvas, reportes y JSON registrados como artefactos.
- [x] Pipeline completo registrado como modelo con firma e input example.
- [x] Corridas agrupadas en `Bank_Marketing_Experimentacion`.

## 4. Corridas y comparación — 20%

- [x] Se definieron seis configuraciones diferentes.
- [ ] Insertar captura legible de la vista Compare de MLflow.
- [ ] Insertar captura del detalle de la mejor corrida y sus artefactos.

## 5. Reproducibilidad — 10%

- [x] Semilla aplicada en Python, NumPy, split y estimador.
- [x] Versiones fijadas en `requirements.txt`.
- [x] README con comandos exactos.
- [ ] Verificar dos corridas idénticas en el equipo de entrega.

## 6. Análisis — 10%

- [x] Mejor configuración justificada con métricas adecuadas al desbalance.
- [x] Hiperparámetro más influyente identificado.
- [x] Limitaciones y mejoras reconocidas.

## Pendientes externos

- [ ] Crear el repositorio individual de GitHub y agregar su enlace.
- [ ] Ejecutar las corridas en el entorno local definitivo.
- [ ] Tomar las dos capturas solicitadas.
- [ ] Integrar análisis, comandos, enlace y capturas en el PDF final.
