# PHM08 · Estimación de RUL con CNN-LSTM

Proyecto final de **MT-9002 Mantenimiento de Sistemas Mecatrónicos** (TEC, II-2026).
Objetivo: estimar la vida útil remanente (**RUL**, en ciclos) de motores turbofan del reto **PHM08** (NASA C-MAPSS), replicando un modelo CNN-LSTM basado en Zhou et al. 2020 (arXiv 2008.03961).

La implementación va por etapas: primero una **CNN 1D** de línea base, luego se reemplaza la cabeza densa por una **LSTM**.

## Estructura

```
.
├── data/                     Datos PHM08 (train.txt, test.txt) + readme oficial
├── src/
│   ├── cnn_rul.py            CNN 1D de línea base (PyTorch)
│   └── ablation_sensores.py  Ablación de sensores s6/s16 con 3 semillas
├── docs/
│   ├── README_paso_a_paso.md            Bitácora técnica por pasos
│   ├── README_productos.md              Inventario de entregables
│   ├── README_IA.md                     Explicación desde la IA
│   └── README_mantenimiento_predictivo.md  Explicación desde el mantenimiento
├── requirements.txt
└── .gitignore
```

## Estado

| Paso | Descripción | Estado |
|---|---|---|
| 0 | Datos PHM08 | ✅ |
| 1 | Elección del modelo y artículo | ⚠️ falta PDF y aprobación del profesor |
| 2 | Preprocesamiento (`prep_cnn_lstm.py` → `phm08_cnn_lstm.npz`) | ⚠️ script no incluido aún |
| 3 | CNN 1D de línea base (`src/cnn_rul.py`) | ✅ código · ⚠️ resultados sin registrar |
| 4 | Ablación de sensores (`src/ablation_sensores.py`) | ✅ código · ⚠️ resultados sin registrar |
| 5 | CNN-LSTM | ⏳ |
| 6 | Evaluación final, informe IEEE y presentación | ⏳ |

Detalle en [`docs/README_paso_a_paso.md`](docs/README_paso_a_paso.md).

## Uso

```bash
pip install -r requirements.txt

# 1) Generar el .npz (script pendiente de agregar al repo)
#    python prep_cnn_lstm.py   ->  src/phm08_cnn_lstm.npz

# 2) Entrenar la CNN de línea base
python src/cnn_rul.py             # -> src/cnn_best.pt, src/pred_test_cnn.txt

# 3) Ablación de sensores
python src/ablation_sensores.py   # -> src/ablation_sensores.csv
```

Los scripts buscan `phm08_cnn_lstm.npz` **en su misma carpeta** (`src/`).

## Modelo actual (CNN 1D)

`(B,30,22) → Conv1d 32 k5 → BN → ReLU → Conv1d 64 k5 → BN → ReLU → MaxPool 2 → Flatten 960 → Dropout 0.3 → Dense 64 → ReLU → Dense 1`

- Entrada: ventanas de 30 ciclos × 22 features (16 sensores con z-score por régimen + 6 one-hot de régimen).
- Etiqueta: RUL saturado en 130.
- Entrenamiento: Adam 3e-4, MSE, batch 512, early stopping (paciencia 15).
- Métricas: RMSE y score asimétrico PHM08 sobre la validación por motor.
- Referencias en la misma validación: predecir la media → RMSE 43,4; regresión lineal → 23,7.

## Datos

PHM08 Prognostics Data Challenge — NASA Ames PCoE (C-MAPSS). Ver [`data/README.md`](data/README.md).

## Referencias

- Zhou et al. 2020, CNN-LSTM para RUL, arXiv 2008.03961.
- Saxena, Goebel, Simon & Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08.
- Ramasso & Saxena, *Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets*, 2014.
- Wang, Yu, Siegel & Lee, *A Similarity-Based Prognostics Approach for RUL Estimation of Engineered Systems*, PHM08.
- EN 62740:2015, Análisis de causa raíz (extracto).
