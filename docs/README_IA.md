# README 3 — Qué está pasando, visto desde la inteligencia artificial

Última actualización: 2026-10-19. Explicación conceptual; los detalles de código están en `README_paso_a_paso.md`.

## 1. El problema como aprendizaje supervisado
Es una **regresión**: a partir de una ventana de los últimos 30 ciclos de un motor (30 × 22 números), predecir un solo número, el RUL. Las etiquetas existen solo en entrenamiento, porque allí cada motor corre hasta fallar y el RUL de cada ciclo se calcula hacia atrás desde la falla.

## 2. Por qué se transforman los datos así
- **Etiqueta saturada en 130.** Al inicio de la vida el motor está sano y los sensores casi no contienen información sobre cuánto falta; pedirle a la red distinguir "falta 300" de "falta 200" con datos indistinguibles solo agrega ruido a la función de pérdida. Saturar dice: "por encima de 130 ciclos, todo es *sano*". Es una **convención** (el 130 es un hiperparámetro, a contrastar con el artículo). Costo: la salida es lineal y no tiene cota, pero el modelo aprende a no predecir mucho más de 130; en motores de test con RUL real mayor, la predicción será sistemáticamente temprana.
- **Normalización por régimen.** Los 3 *settings* forman 6 condiciones de vuelo que cambian el nivel de los sensores mucho más que la degradación (p. ej., en s2 ~106 unidades entre regímenes frente a ~1 unidad por degradación). Restar la media y dividir por la desviación **de cada régimen** deja visible la desviación respecto de la **media de la flota en esa condición** (calculada sobre toda la vida, sana y degradada, de los motores de entrenamiento), no respecto de una línea base sana individual. El *one-hot* de régimen se agrega para que la red sepa en qué condición está cada ciclo.
- **Descartar sensores.** Se quitan los que no varían dentro de ningún régimen (s1, s5, s18, s19): no aportan señal y sí parámetros. s10 y s16 son casos dudosos (señales de 2 niveles, con algo de información); se descartó s10 y se conservó s16 sin justificación registrada. La ablación de s6/s16 verifica empíricamente si dos sensores dudosos ayudan o estorban.
- **Ventanas.** La red recibe un tramo de historia, no un ciclo suelto: la tendencia es más informativa que un valor aislado y ruidoso.

## 3. Qué aprende la CNN
- Una **convolución 1D** desliza filtros de 5 ciclos sobre el tiempo; cada filtro combina los 22 canales a la vez, así que puede detectar patrones conjuntos ("s11 sube mientras s12 baja"). Los mismos pesos se aplican en toda la ventana → las convoluciones tienen pocos parámetros (≈14 000) y detectan el patrón esté donde esté.
- Dos capas apiladas amplían el **campo receptivo** a 9 ciclos y el *max-pooling* a 10, reduciendo el eje temporal de 30 a 15.
- **BatchNorm** estabiliza la escala de las activaciones; **Dropout** (30 %) pone en cero al azar el 30 % de los 960 rasgos aplanados antes de la capa densa, para que no dependa de rasgos concretos (menos memorización).
- La **cabeza densa** aplana (64 × 15 = 960 valores); cada posición temporal tiene pesos propios, así que sí distingue *dónde* en la ventana ocurrió cada patrón.
- Parámetros: 75 617 en total; el 81 % (61 504) está en la primera capa densa `Linear(960→64)`.

## 4. Cómo se entrena y se elige el modelo
- Se minimiza el **MSE** sobre el RUL escalado a [0, 1] con **Adam**.
- **Validación por motor:** los motores de validación no aparecen en entrenamiento. Si se repartieran ventanas al azar, ventanas casi idénticas del mismo motor estarían en ambos lados y el error de validación sería engañosamente bajo (fuga de información).
- ***Early stopping*:** se guarda el modelo de la época con menor RMSE de validación y se para tras 15 épocas sin mejora. La validación se usa para elegir, así que su RMSE es **algo optimista**; lo mismo ocurre al elegir la variante de sensores con esa misma validación. Una estimación limpia requiere datos no usados para ninguna decisión (los del profesor).
- **Semillas múltiples** en la ablación: con una sola semilla una diferencia de 0,5 ciclos puede ser azar. Pero 3 semillas con desviación poblacional subestiman la dispersión; las semillas solo cambian inicialización y orden de batches (no el split), y no forman pares entre variantes con distinto número de entradas.
- **Qué mide el RMSE de validación:** todas las ventanas, contra el RUL saturado; el 29 % de las etiquetas vale 130 (meseta fácil). No es el error en la última ventana de cada motor, que es lo comparable con la literatura.

## 5. Hacia la CNN-LSTM
La cabeza densa sí distingue posiciones, pero no procesa la secuencia en orden ni comparte pesos entre pasos. La **LSTM** recorre los 15 pasos con un estado que se arrastra y comparte pesos en el tiempo; la hipótesis (a confirmar citando la sección correspondiente de Zhou et al. 2020, cuyo PDF aún no está en el proyecto) es que ese sesgo inductivo secuencial modela mejor la evolución de la degradación. Además reemplaza la capa que concentra el 81 % de los parámetros. Se probará comparando contra la CNN con el mismo split, semillas y métricas.

## 6. Métricas
- **RMSE**: simétrico, en ciclos; penaliza errores grandes.
- ***Score* PHM08**: exponencial y **asimétrico** — predecir más vida de la real (tarde) se castiga más (divisor 10) que predecir menos (divisor 13). Es una suma, crece con el número de unidades. El entrenamiento optimiza MSE (simétrico), así que el modelo no está sesgado hacia predicciones tempranas; limitación conocida.

## 7. Límites actuales
- Una sola partición train/validación: el resultado depende de qué motores cayeron en validación.
- Métrica de validación por ventana y contra RUL saturado (ver §4).
- Resultados numéricos de la CNN y de la ablación aún no registrados; sin meta numérica del artículo.
- Manejo de motores de test con < 30 ciclos y de la desviación 0 de s16 no documentados.
