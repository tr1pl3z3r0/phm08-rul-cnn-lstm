"""
ablation_sensores.py - Compara la CNN quitando sensores (s6, s16) con varias semillas.

No hace falta volver a correr prep_cnn_lstm.py: la normalización es columna por columna
y el split por motor no depende de los sensores, así que quitar la columna de un sensor
del .npz da exactamente los mismos datos que regenerarlo con ese sensor en DROP.
(Para AGREGAR un sensor que no está en el .npz, como s10, sí hay que regenerarlo en prep.)

Modelo y entrenamiento: los mismos de cnn_rul.py.
Salida: tabla en pantalla y ablation_sensores.csv en esta carpeta.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- CONFIGURACIÓN ----------------
NPZ_FILE = "phm08_cnn_lstm.npz"
# Orden de las columnas en el .npz (DROP = s1, s5, s10, s18, s19) + 6 one-hot de régimen
SENSORS = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s11", "s12",
           "s13", "s14", "s15", "s16", "s17", "s20", "s21"]
VARIANTES = {                      # nombre: sensores que se quitan
    "todos (actual)": [],
    "sin s6": ["s6"],
    "sin s16": ["s16"],
    "sin s6 ni s16": ["s6", "s16"],
}
SEEDS = (0, 1, 2)
# Mismos hiperparámetros que cnn_rul.py
RUL_MAX, CONV_CH, KERNEL, POOL, HIDDEN, DROPOUT = 130, (32, 64), 5, 2, 64, 0.3
LR, BATCH, EPOCHS, PATIENCE = 3e-4, 512, 100, 15
# ------------------------------------------------

HERE = Path(__file__).resolve().parent
device = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(HERE / NPZ_FILE)
assert d["X_train"].shape[2] == len(SENSORS) + 6, "el .npz no tiene el orden de columnas esperado"


def build(n_feat, window):
    c1, c2 = CONV_CH
    return nn.Sequential(
        nn.Conv1d(n_feat, c1, KERNEL, padding=KERNEL // 2), nn.BatchNorm1d(c1), nn.ReLU(),
        nn.Conv1d(c1, c2, KERNEL, padding=KERNEL // 2), nn.BatchNorm1d(c2), nn.ReLU(),
        nn.MaxPool1d(POOL),
        nn.Flatten(), nn.Dropout(DROPOUT),
        nn.Linear(c2 * (window // POOL), HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1),
    ).to(device)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = [model(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(out).squeeze(1).numpy() * RUL_MAX


def phm_score(y, p):
    e = p.astype(np.float64) - y
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))


def entrenar(cols, seed):
    """Entrena una CNN con las columnas `cols`; devuelve (RMSE val, score val, mejor época)."""
    torch.manual_seed(seed)
    # Se transpone una sola vez a (N, F, W), que es lo que espera Conv1d
    Xtr = torch.from_numpy(d["X_train"][:, :, cols].transpose(0, 2, 1).copy())
    Xva = torch.from_numpy(d["X_val"][:, :, cols].transpose(0, 2, 1).copy())
    ytr, yva = d["y_train"], d["y_val"]
    dl = DataLoader(TensorDataset(Xtr, torch.from_numpy(ytr / RUL_MAX)), batch_size=BATCH, shuffle=True)
    model = build(len(cols), Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, wait = (np.inf, None, 0), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        p = predict(model, Xva)
        r = float(np.sqrt(np.mean((p - yva) ** 2)))
        if r < best[0]:
            best, wait = (r, phm_score(yva, p), epoch), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best


filas = []
for nombre, quitar in VARIANTES.items():
    cols = [i for i, s in enumerate(SENSORS) if s not in quitar] + list(range(len(SENSORS), len(SENSORS) + 6))
    res = []
    for seed in SEEDS:
        r, s, ep = entrenar(cols, seed)
        res.append((r, s))
        print(f"{nombre:<16} semilla {seed}: RMSE {r:6.2f}  score {s:8.0f}  (época {ep})", flush=True)
    R, S = np.array(res).T
    filas.append((nombre, len(cols), R.mean(), R.std(), S.mean(), S.std()))

print(f"\n{'variante':<16}{'F':>4}{'RMSE val':>18}{'score val':>22}")
for n, f, rm, rs, sm, ss in filas:
    print(f"{n:<16}{f:>4}{rm:>11.2f} ± {rs:<5.2f}{sm:>13.0f} ± {ss:<6.0f}")
np.savetxt(HERE / "ablation_sensores.csv",
           np.array([[f, rm, rs, sm, ss] for _, f, rm, rs, sm, ss in filas]),
           delimiter=",", fmt="%.4f", header="F,rmse_media,rmse_std,score_media,score_std",
           comments="# variantes: " + " | ".join(VARIANTES) + "\n")
print("Guardado: ablation_sensores.csv")
