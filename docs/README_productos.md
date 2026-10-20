# README 2 — Productos finales de cada paso

Solo los entregables: qué es, quién lo genera, qué contiene y si está guardado en el proyecto. Para el *cómo*, ver `README_paso_a_paso.md`.

Última actualización: 2026-10-19. ⚠️ = inferido de comentarios de `cnn_rul.py`; por confirmar con `prep_cnn_lstm.py`.

| # | Producto | Lo genera | Contenido / formato | ¿En el proyecto? |
|---|---|---|---|---|
| 0 | `train.txt`, `test.txt` | NASA Ames PCoE (C-MAPSS), publicado para el reto PHM08; vía zip del curso | Texto, 26 columnas; 218 motores cada uno (distintos) | ✅ `data/` |
| 2 | `prep_cnn_lstm.py` | Grupo | Script de preprocesamiento | ❌ **falta subirlo** |
| 2 | `phm08_cnn_lstm.npz` ⚠️ | `prep_cnn_lstm.py` | `X_train` (32 057, 30, 22), `y_train` (32 057,), `X_val` (7 539, 30, 22), `y_val` (7 539,), `X_test` (218, 30, 22). 22 = 16 sensores normalizados por régimen + 6 *one-hot* de régimen; y = RUL saturado en 130. `X_test` con 218 filas implica algún relleno para los 6 motores con < 30 ciclos (no documentado) | ❌ (archivo binario local) |
| 3 | `cnn_rul.py` | Grupo | CNN 1D de línea base, en PyTorch | ✅ `src/cnn_rul.py` |
| 3 | `cnn_best.pt` | `cnn_rul.py` | Pesos del modelo con menor RMSE de validación | ❌ (local) |
| 3 | `pred_test_cnn.txt` | `cnn_rul.py` | 218 líneas, un RUL predicho por motor de test (ciclos, 2 decimales) | ❌ (local) |
| 3 | RMSE y *score* de validación de la CNN | consola de `cnn_rul.py` | Números | ❌ **no registrados** |
| 4 | `ablation_sensores.py` | Grupo | Ablación de s6 / s16 con 3 semillas | ✅ `src/ablation_sensores.py` |
| 4 | `ablation_sensores.csv` | `ablation_sensores.py` | Una fila por variante: F, RMSE media, RMSE desv. (poblacional), *score* media, *score* desv. | ❌ (local) |
| 5 | Modelo CNN-LSTM | — | — | ⏳ pendiente |
| 6 | Informe IEEE + presentación | — | — | ⏳ pendiente |

## Cómo leer cada producto

- **`pred_test_cnn.txt`** ⚠️: la línea *i* sería el RUL estimado del motor *i* de `test.txt` después de su último ciclo observado (orden por confirmar en `prep`). No se puede calcular RMSE localmente porque el RUL real de test no es público.
- **RMSE de validación**: error en ciclos sobre **todas las ventanas** de los motores de validación, medido contra el RUL **saturado en 130** (29 % de las etiquetas son exactamente 130). **No** equivale al error en la última ventana de cada motor ni es comparable con RMSE publicados por motor. Comparar contra 43,4 (predecir la media) y 23,7 (regresión lineal), calculados igual.
- ***Score* PHM08 de validación**: suma de penalizaciones exponenciales sobre 7 539 ventanas. Solo sirve para comparar modelos sobre **esta misma** validación. La tabla de *scores* del readme oficial (436–2 430) probablemente es del *leaderboard* sobre `test.txt` (218 motores; ver Ramasso & Saxena 2014, Tabla 2); el ranking final usó `final_test.txt` (435 motores, *scores* desde 5 636). Como el *score* es una suma, solo se compara con el mismo número de unidades — también con los datos del profesor.
- **`ablation_sensores.csv`**: criterio propuesto para preferir una variante: diferencia de medias > 2 × desviación **muestral** (ddof = 1, recalculada a partir de las corridas). Con 3 semillas este criterio es débil; idealmente 5+ semillas o validación cruzada por motor.

## Para completar este inventario
Subir al proyecto `prep_cnn_lstm.py`, el PDF de Zhou et al. 2020, y pegar la salida de consola de `cnn_rul.py` y de `ablation_sensores.py` (o el `.csv`).
