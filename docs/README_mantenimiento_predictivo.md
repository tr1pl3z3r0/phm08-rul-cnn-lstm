# README 4 — Qué está pasando, visto desde el mantenimiento predictivo

Última actualización: 2026-10-19. Explicación conceptual para el curso MT-9002.

## 1. El activo y su modo de falla
PHM08 contiene trayectorias de motores turbofan del mismo tipo simuladas con C-MAPSS (NASA Ames): **218 motores de entrenamiento y 218 de prueba, todos distintos** (el reto tenía además 435 en `final_test`). Cada motor arranca con un desgaste inicial y una variación de fabricación distintos (normales, no son falla).

**Modo de falla:** según Saxena et al. 2008, la degradación se simula como **pérdida exponencial de flujo y eficiencia de los módulos rotativos** (fan, LPC, HPC, HPT, LPT), a partir de un punto inicial distinto en cada motor. El fin de vida (F) ocurre cuando el **índice de salud** —el mínimo de los márgenes de operación— llega a cero. Ramasso & Saxena (2014) clasifican PHM08 como **1 modo de falla y 6 condiciones de operación**.

Cómo se evidencia: los 21 sensores (temperaturas, presiones, velocidades, flujos) registran el efecto de esa pérdida de eficiencia como tendencias lentas, mezcladas con **ruido de medición** y con el efecto, mucho mayor, de las **6 condiciones de operación**.

## 2. De mantenimiento correctivo/preventivo a predictivo
- **Correctivo:** se repara cuando falla → paradas no planificadas.
- **Preventivo por tiempo:** se interviene cada N ciclos → se desperdicia vida útil de los motores sanos.
- **Predictivo (lo que hace este proyecto):** se estima, con los datos de condición de *cada* motor, cuántos ciclos le quedan (**RUL**), y se planifica la intervención antes del fin de su vida útil.

En la curva P-F, el punto **P** (falla potencial) es cuando la degradación se hace visible y **F** la falla funcional. El modelo **no detecta P explícitamente**: la saturación del RUL en 130 **supone** que P está a ~130 ciclos de F en todos los motores, aunque en realidad el inicio de la degradación varía entre motores. Es una convención de modelado (RUL lineal a trozos) que debe justificarse con el artículo o la literatura.

## 3. Cómo se traduce cada paso técnico
| Paso técnico | Significado en mantenimiento |
|---|---|
| Normalizar por régimen | Comparar cada lectura con la **media de la flota en esa condición de operación** (calculada sobre toda la vida, sana y degradada, de los motores de entrenamiento). Se parece a comparar una vibración a la misma carga, pero no es una línea base sana individual. |
| Descartar sensores | No todos los puntos de medición son indicadores de condición: se quitan los que no varían dentro de ningún régimen. Dos sensores de 2 niveles (s10, s16) son casos límite aún sin justificar. |
| Saturar el RUL en 130 | Convención: en la zona sana no hay síntomas útiles; el pronóstico se concentra en la zona de degradación. |
| Ventana de 30 ciclos | El diagnóstico se basa en la **tendencia** reciente, no en una lectura aislada. |
| Validar con motores no vistos | Simula la situación real: el modelo se aplicará a motores nuevos. |
| *Score* asimétrico | Predecir que queda **más** vida de la real (tarde) puede llevar a una falla en servicio; predecir menos (temprano) solo adelanta una intervención. Por eso tardar se castiga más. |

## 4. Uso esperado del resultado
Con `pred_test_cnn.txt` (o la salida futura de la CNN-LSTM) se obtiene un RUL por motor. Un planificador lo usaría para:
1. Ordenar la flota por urgencia.
2. Programar la intervención con un margen de seguridad.
3. Agrupar intervenciones de motores con RUL similar para reducir paradas.

**Margen de seguridad:** el RMSE de validación **no sirve** como margen: es simétrico, no da un nivel de confianza, no distingue errores tardíos de tempranos y hoy se mide sobre todas las ventanas contra el RUL saturado. Propuesta: margen = **percentil 90–95 del error tardío** (predicho − real > 0) en la zona de RUL bajo de validación (p. ej., RUL real ≤ 50), medido en la última ventana o en ventanas cercanas a la falla. Ese cálculo **aún no se ha hecho**.

## 5. Relación con el análisis de causa raíz (norma EN 62740)
El proyecto incluye un extracto de la norma EN 62740:2015 de RCA. El modelo de RUL responde **cuándo** fallará un motor, no **por qué**. Complementa el RCA: el pronóstico permite intervenir antes del evento foco, y el RCA explica las causas cuando el evento ocurre. La ablación de sensores da una pista indirecta de qué variables físicas reflejan la degradación, pero no identifica causas.

## 6. Limitaciones desde el punto de vista del mantenimiento
- Datos **simulados** con un solo modo de falla (Ramasso & Saxena 2014); en planta hay múltiples modos y fallas súbitas que este modelo no captaría.
- El modelo no entrega **incertidumbre** (intervalo de confianza), solo un número.
- El techo de 130 ciclos impide planificar a muy largo plazo y produce predicciones tempranas en motores con mucha vida restante.
- Se requieren 30 ciclos de historia por motor para la ventana completa; 6 motores de test tienen menos, y su tratamiento no está documentado.
