# Datos — PHM08 Prognostics Data Challenge

Generados por NASA Ames PCoE con el simulador C-MAPSS y publicados para el reto PHM08. Readme oficial: [`phm08_readme.txt`](phm08_readme.txt).

| Archivo | Motores | Filas | Descripción |
|---|---|---|---|
| `train.txt` | 218 | 45 918 | Cada motor corre hasta la falla. Largo 128–357 ciclos. |
| `test.txt` | 218 (distintos) | 29 820 | Series cortadas antes de la falla; RUL real no publicado. Largo 15–364 ciclos; 6 motores con < 30 ciclos. |

## Formato

Texto separado por espacios, una fila por ciclo, 26 columnas:

| Col. | Contenido |
|---|---|
| 1 | Número de unidad (motor) |
| 2 | Ciclo |
| 3–5 | Settings operativos 1–3 (forman 6 regímenes) |
| 6–26 | Sensores s1–s21 |

> El readme oficial menciona "sensor 26" por error; los datos tienen 21 sensores.
