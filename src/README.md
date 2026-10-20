# Código

| Script | Qué hace | Entrada | Salida |
|---|---|---|---|
| `cnn_rul.py` | Entrena la CNN 1D de línea base y predice el RUL de los 218 motores de test | `phm08_cnn_lstm.npz` | `cnn_best.pt`, `pred_test_cnn.txt` |
| `ablation_sensores.py` | Compara la CNN quitando s6 / s16 (4 variantes × 3 semillas) | `phm08_cnn_lstm.npz` | `ablation_sensores.csv` |

Ambos esperan `phm08_cnn_lstm.npz` en esta carpeta. Lo genera `prep_cnn_lstm.py` (pendiente de agregar), con:

- `X_train` (32 057, 30, 22), `y_train`
- `X_val` (7 539, 30, 22), `y_val`
- `X_test` (218, 30, 22)

22 features = 16 sensores (z-score por régimen) + 6 one-hot de régimen; `y` = RUL saturado en 130.

Pendiente: `cnn_lstm_rul.py`, que reutiliza `CNNExtractor` de `cnn_rul.py` y reemplaza la cabeza densa por una LSTM sobre los 15 pasos de tiempo.
