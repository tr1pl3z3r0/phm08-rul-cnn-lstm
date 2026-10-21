"""Preparación de datos PHM08 para CNN-LSTM.
Salida: X_train (N, W, F), y_train (N,), X_val, y_val, X_test (218, W, F)
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Rutas relativas a la carpeta del script (funciona desde cualquier terminal)
DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = os.path.join(DIR, "train.txt"), os.path.join(DIR, "test.txt")
W = 30            # largo de ventana (ciclos)
RUL_MAX = 130     # RUL lineal por tramos (Heimes 2008)
VAL_FRAC = 0.2
SEED = 0

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
# ---- SELECCIÓN DE SENSORES: agregar o quitar de esta lista ----
# s1, s5, s18, s19: constantes dentro de cada régimen (no aportan nada)
# s10: casi constante, correlación ~0 con el RUL
# s6, s16: casi constantes pero con algo de información -> probar con y sin
DROP = ["s1", "s5", "s10", "s18", "s19"]
SENSORS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]


def load(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


tr, te = load(TRAIN), load(TEST)

# 1) Etiqueta RUL (solo train): ciclos restantes, saturada en RUL_MAX
tr["RUL"] = tr.groupby("unit")["cycle"].transform("max") - tr["cycle"]
tr["RUL"] = tr["RUL"].clip(upper=RUL_MAX)

# 2) Régimen de operación: 6 condiciones a partir de os1-os3
km = KMeans(n_clusters=6, n_init=10, random_state=SEED)
km.fit(tr[["os1", "os2", "os3"]])
tr["reg"] = km.predict(tr[["os1", "os2", "os3"]])
te["reg"] = km.predict(te[["os1", "os2", "os3"]])

# 3) Split por motor (nunca por filas, evita fuga de información)
rng = np.random.default_rng(SEED)
units = tr["unit"].unique()
val_units = rng.choice(units, int(len(units) * VAL_FRAC), replace=False)
is_val = tr["unit"].isin(val_units)

# 4) Normalización z-score por régimen, con estadísticas SOLO de train
stats = tr[~is_val].groupby("reg")[SENSORS].agg(["mean", "std"])
def normalize(df):
    df = df.copy()
    df[SENSORS] = df[SENSORS].astype(float)
    for r in range(6):
        m = df["reg"] == r
        mu = stats.loc[r, (slice(None), "mean")].values
        sd = stats.loc[r, (slice(None), "std")].values
        sd = np.where(sd > 1e-6, sd, 1.0)   # sensor constante -> no dividir entre ~0
        df.loc[m, SENSORS] = (df.loc[m, SENSORS].values - mu) / sd
    return df
tr, te = normalize(tr), normalize(te)

# 5) Features: sensores normalizados + régimen one-hot
FEATS = SENSORS + [f"reg_{r}" for r in range(6)]
for df in (tr, te):
    for r in range(6):
        df[f"reg_{r}"] = (df["reg"] == r).astype(float)


def pad(a):
    """Si la serie es más corta que W, repite la primera fila al inicio."""
    if len(a) < W:
        a = np.vstack([np.repeat(a[:1], W - len(a), axis=0), a])
    return a


def windows(df, with_y=True):
    X, y = [], []
    for _, g in df.groupby("unit"):
        a = pad(g[FEATS].values)
        r = pad(g[["RUL"]].values)[:, 0] if with_y else None
        for i in range(len(a) - W + 1):
            X.append(a[i:i + W])
            if with_y:
                y.append(r[i + W - 1])      # RUL en el último ciclo de la ventana
    return np.array(X, np.float32), (np.array(y, np.float32) if with_y else None)


X_train, y_train = windows(tr[~is_val])
X_val, y_val = windows(tr[is_val])

# 6) Test: solo la ÚLTIMA ventana de cada motor -> una predicción de RUL por motor
X_test = np.array([pad(g[FEATS].values)[-W:] for _, g in te.groupby("unit")],
                  np.float32)

print("X_train", X_train.shape, "X_val", X_val.shape, "X_test", X_test.shape)
np.savez(os.path.join(DIR, "phm08_cnn_lstm.npz"), X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val, X_test=X_test)


def phm_score(y_true, y_pred):
    """Score asimétrico PHM08 (penaliza más las predicciones tardías)."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))