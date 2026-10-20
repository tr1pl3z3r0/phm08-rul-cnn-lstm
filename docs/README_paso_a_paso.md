# README 1 — Paso a paso del proyecto (bitácora técnica)

Proyecto final MT-9002 (Mantenimiento de Sistemas Mecatrónicos, TEC, II-2026): estimar la vida útil remanente (RUL) de los motores turbofan del reto **PHM08**. La guía del curso pide **replicar** el modelo de un artículo; el artículo elegido es Zhou et al. 2020 (*ECLSTM*, CNN-LSTM, arXiv 2008.03961).

Última actualización: 2026-10-19. Fuentes revisadas: `src/cnn_rul.py` (17-oct), `src/ablation_sensores.py` (18-oct), el zip del curso (datos, guía, Saxena et al. 2008, Ramasso & Saxena 2014) y la memoria del proyecto.

> **Convención de estado.** ✅ hecho y verificable en el proyecto · ⚠️ hecho pero el archivo, la evidencia o el resultado **no está** en el proyecto (se infiere de comentarios del código) · ⏳ pendiente.

---

## Paso 0 — Datos de partida ✅

| Archivo | Contenido verificado |
|---|---|
| `train.txt` | 218 motores, 45 918 filas (una por ciclo), 26 columnas: unidad, ciclo, 3 *settings* operativos, 21 sensores. Cada motor corre **hasta la falla**. Largo: 128–357 ciclos (media 210,6). |
| `test.txt` | 218 motores **distintos** de los de train, 29 820 filas. Cada serie se **corta antes** de la falla; el RUL real no se publica. Largo: 15–364 ciclos; **6 motores tienen menos de 30 ciclos** (15, 16, 23, 25, 26, 28). |
| `final_test.txt` | Mencionado en el readme oficial (435 motores, ranking final del reto). **No está** en el zip del curso. |
| Guía del curso | Pide replicar el modelo del artículo, intentar alcanzar sus resultados, validar y competir con datos que entrega el profesor. |

Origen de los datos: simulador C-MAPSS de NASA Ames (PCoE), publicado para el reto PHM08.

## Paso 1 — Elección del modelo y del artículo ⚠️

Modelo CNN-LSTM; artículo de referencia Zhou et al. 2020. Decisión de implementarlo **por etapas**: primero una CNN sola (línea base), luego reemplazar la cabeza densa por una LSTM.
Pendiente: subir el PDF del artículo al proyecto y registrar la fecha de aprobación del profesor (la guía, punto c, la exige antes de implementar).

### Artículo vs. implementación (a completar con el PDF)

| Elemento | Artículo (Zhou et al. 2020) | Implementación actual | ¿Igual o decisión propia? |
|---|---|---|---|
| Arquitectura | por completar | CNN 1D (2 conv) + densa; LSTM pendiente | por completar |
| Ventana W | por completar | 30 | por completar |
| Tope de RUL | por completar | 130 | por completar |
| Normalización | por completar | z-score por régimen + one-hot | por completar |
| Kernel / filtros / dropout | por completar | 5 / 32-64 / 0,3 | por completar |
| Conjunto de evaluación | por completar | validación por motor (19 %) | por completar |
| Resultado reportado (meta) | **por completar** — sin esta cifra el Paso 6 no se puede evaluar | — | — |

## Paso 2 — Preparación de datos (`prep_cnn_lstm.py`) ⚠️

El script **no está guardado en el proyecto**; lo que sigue se deduce de los comentarios y *asserts* de los scripts que sí están:

1. **Selección de sensores:** se descartan s1, s5, s10, s18, s19 → quedan 16 sensores (s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s16, s17, s20, s21).
   - s1, s5, s18 y s19 tienen desviación **nula dentro de cada régimen** (solo varían porque cambia el régimen).
   - s10 y s16 son señales **cuantizadas de 2 niveles** que solo varían en algunos regímenes (s16 en 1 de 6, s10 en 4 de 6) y ambas llevan algo de información de degradación. Se descartó s10 y se conservó s16; **la razón no está documentada** (decisión pendiente de justificar; probar "con s10" exige regenerar el `.npz`).
2. **Regímenes de operación:** los 3 *settings* redondeados forman exactamente 6 regímenes; cada uno se codifica *one-hot* (6 columnas).
3. **Normalización:** *z-score* de cada sensor **dentro de su régimen**, con media y desviación de toda la vida (sana y degradada) de los motores. Por confirmar: que se ajuste solo con train, y cómo se evita dividir entre 0 en s16 (constante en 5 regímenes).
4. **Etiqueta RUL:** RUL = último ciclo − ciclo actual, **saturado en 130** (RUL lineal a trozos). El 130 es un hiperparámetro (ver tabla del Paso 1).
5. **Ventanas deslizantes:** W = 30 ciclos, paso 1; la etiqueta es el RUL del último ciclo de la ventana.
6. **Split por motor** (no por ventana) en train/validación. Comprobación: Σ(largo − 29) = 39 596 ventanas = 32 057 (train) + 7 539 (val), ≈ 81/19.
7. **Test:** una ventana (la última) por motor → 218 ventanas.
8. Salida: `phm08_cnn_lstm.npz` con `X_train, y_train, X_val, y_val, X_test`; forma de X = (n, 30, 22).

Puntos abiertos: manejo de los 6 motores de test con < 30 ciclos (¿relleno?), datos usados para ajustar la normalización, división entre 0 en s16, semilla del split.

## Paso 3 — CNN 1D de línea base (`cnn_rul.py`) ✅ (código) / ⚠️ (resultados)

- Arquitectura: `Conv1d(22→32, k=5) → BN → ReLU → Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2) → Flatten(960) → Dropout 0,3 → Dense 64 → ReLU → Dense 1`. 75 617 parámetros.
- Entrenamiento: Adam (lr 3e-4), MSE sobre RUL/130, batch 512, hasta 100 épocas, *early stopping* con paciencia 15 sobre el RMSE de validación, semilla 0.
- Métricas: RMSE (ciclos) y *score* asimétrico PHM08, sobre **todas** las ventanas de validación y contra el RUL **saturado** (29 % de esas etiquetas valen exactamente 130).
- Referencias en esa misma validación: predecir la media → RMSE 43,4 (= desviación estándar de las etiquetas); regresión lineal → RMSE 23,7.
- Salidas: `cnn_best.pt` (pesos) y `pred_test_cnn.txt` (218 RUL).
- **El RMSE de validación alcanzado por la CNN no está registrado en el proyecto.**
- Mejora recomendada: agregar (1) RMSE y *score* en la **última ventana de cada motor de validación** (comparable con el artículo) y (2) RMSE en la zona RUL ≤ 50 (útil para mantenimiento).

## Paso 4 — Ablación de sensores (`ablation_sensores.py`) ✅ (código) / ⚠️ (resultados)

- Pregunta: ¿s6 y s16 aportan o meten ruido?
- Variantes: todos / sin s6 / sin s16 / sin s6 ni s16; 3 semillas cada una (12 entrenamientos), misma CNN y mismos hiperparámetros.
- Se quitan columnas del `.npz` en vez de regenerarlo (válido porque la normalización es por columna y el split no depende de los sensores).
- Salida: tabla media ± desviación de RMSE y *score*, y `ablation_sensores.csv`.
- Limitaciones: 3 semillas con desviación poblacional (ddof = 0) subestiman la dispersión; las semillas no varían el split; elegir variante con la misma validación agrega optimismo.
- **La tabla resultante y la decisión tomada (¿se quitan s6/s16?) no están registradas en el proyecto.**

## Paso 5 — CNN-LSTM ⏳

Reutilizar `CNNExtractor` (salida (B, 64, 15)), transponer a (B, 15, 64) y pasarlo por una LSTM que recorra los 15 pasos; cabeza densa sobre el último estado oculto. Reemplaza justamente la capa `Linear(960→64)`, que hoy concentra el 81 % de los parámetros. Comparar contra la CNN con el mismo split, semillas y métricas.

## Paso 6 — Evaluación final, informe IEEE y presentación ⏳

Predecir sobre los datos que entregue el profesor, comparar con la meta numérica del artículo (tabla del Paso 1) y redactar el informe en formato IEEE (≤ 10 min de presentación).

---

### Registro de revisiones
- 2026-10-19 — Primera versión (tarea programada) + correcciones C-01, C-02, C-04, C-06, C-07, C-08, C-15 del examinador aplicadas.
