"""
cnn_rul.py - CNN 1D para estimar el RUL de los motores PHM08 (PyTorch).

Entrada: phm08_cnn_lstm.npz (lo genera prep_cnn_lstm.py; debe estar en esta carpeta)
    X_*: (muestras, W=30 ciclos, F=22 features) = 16 sensores (z-score por régimen) + 6 one-hot de régimen
    y_*: RUL del último ciclo de cada ventana, saturado en 130
Salida (en esta carpeta):
    cnn_best.pt        pesos del mejor modelo (menor RMSE de validación)
    pred_test_cnn.txt  un RUL predicho por motor de test (218 líneas)

Estructura pensada para agregar la LSTM después:
    CNNExtractor  convoluciones; conserva el eje del tiempo    (B, 30, 22) -> (B, 64, 15)
    CNN           extractor + cabeza densa que aplana           (B, 64, 15) -> (B,)
    La CNN-LSTM reutiliza CNNExtractor y cambia la cabeza densa por una LSTM
    que recorre los 15 pasos de tiempo.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- CONFIGURACIÓN ----------------
NPZ_FILE = "phm08_cnn_lstm.npz"  # nombre que genera prep_cnn_lstm.py
RUL_MAX = 130         # mismo tope que en prep_cnn_lstm.py (para escalar y a [0, 1])
CONV_CH = (32, 64)    # filtros (canales de salida) de la 1.ª y 2.ª convolución
KERNEL = 5            # largo del kernel en el eje del tiempo (ciclos)
POOL = 2              # max-pooling: divide el largo temporal entre 2 (30 -> 15)
HIDDEN = 64           # neuronas de la capa densa
DROPOUT = 0.3         # fracción de neuronas que se apagan al azar al entrenar
LR = 3e-4             # tasa de aprendizaje de Adam
BATCH = 512           # ventanas por paso de gradiente
EPOCHS = 100          # máximo de épocas
PATIENCE = 15         # early stopping: épocas seguidas sin mejorar en validación
SEED = 0
# ------------------------------------------------

HERE = Path(__file__).resolve().parent
torch.manual_seed(SEED)  # fija la inicialización de pesos y el orden de los batches
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============ 1) DATOS ============
d = np.load(HERE / NPZ_FILE)
X_train, y_train = d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]
X_test = d["X_test"]
N, W, F = X_train.shape  # (32057, 30, 22); F cambia solo si se modifica DROP en prep

# El objetivo se escala a [0, 1]: la última neurona no tiene que "llegar" a 130
# y el MSE queda en un rango parecido al de las entradas. Se des-escala al predecir.
train_dl = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train / RUL_MAX)),
    batch_size=BATCH,
    shuffle=True,  # mezclar ventanas sí es válido: el split por motor ya se hizo en prep
)


# ============ 2) MODELO ============
class CNNExtractor(nn.Module):
    """Convoluciones 1D sobre el tiempo, con cada feature como canal de entrada.

    Cada uno de los 32 kernels de la 1.ª capa mira 5 ciclos seguidos de los 22 canales
    a la vez, así que puede aprender patrones como "s11 sube mientras s12 baja".
    Tamaño de salida (d2l 7.3): L_out = floor((L + 2p - k) / s) + 1
    Con k=5, p=2, s=1: (30 + 4 - 5) / 1 + 1 = 30 -> padding 'same', el largo no cambia.
    Campo receptivo: 5 ciclos tras conv1, 9 tras conv2, 10 tras el pooling.
    BatchNorm1d (d2l 8.5) normaliza cada canal con la media y desviación del batch;
    estabiliza el entrenamiento y en este problema bajó el RMSE de validación.
    """

    def __init__(self, n_feat):
        super().__init__()
        c1, c2 = CONV_CH
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, c1, KERNEL, padding=KERNEL // 2),  # (B, 22, 30) -> (B, 32, 30)
            nn.BatchNorm1d(c1),                                  # misma forma
            nn.ReLU(),
            nn.Conv1d(c1, c2, KERNEL, padding=KERNEL // 2),      # (B, 32, 30) -> (B, 64, 30)
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(POOL),                                  # (B, 64, 30) -> (B, 64, 15)
        )

    def forward(self, x):          # x: (B, W, F), tal como viene del npz
        x = x.transpose(1, 2)      # (B, F, W): Conv1d desliza el kernel sobre el último eje
        return self.net(x)         # (B, 64, W // POOL)


class CNN(nn.Module):
    """Extractor + cabeza densa. Aplanar conserva en qué parte de la ventana
    apareció cada patrón (no es lo mismo al inicio que al final de la ventana)."""

    def __init__(self, n_feat, window):
        super().__init__()
        self.features = CNNExtractor(n_feat)
        L = window // POOL                             # largo temporal tras el pooling: 15
        self.head = nn.Sequential(
            nn.Flatten(),                              # (B, 64, 15) -> (B, 960)
            nn.Dropout(DROPOUT),
            nn.Linear(CONV_CH[-1] * L, HIDDEN),        # (B, 960) -> (B, 64)
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),                      # (B, 64) -> (B, 1): RUL escalado
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)  # (B, 1) -> (B,)


model = CNN(F, W).to(device)

# Recorrido de formas con un batch, capa por capa (mismo orden que forward)
model.eval()  # para que BatchNorm no actualice sus estadísticas con este batch de prueba
with torch.no_grad():
    x = torch.from_numpy(X_train[:BATCH]).to(device)
    print(f"{'entrada (B, W, F)':<24}{tuple(x.shape)}")
    x = x.transpose(1, 2)
    print(f"{'transpose (B, F, W)':<24}{tuple(x.shape)}")
    for layer in [*model.features.net, *model.head]:
        x = layer(x)
        print(f"{type(layer).__name__:<24}{tuple(x.shape)}")
print("parámetros entrenables:", sum(p.numel() for p in model.parameters()))
print("dispositivo:", device, "\n")


# ============ 3) MÉTRICAS ============
def predict(X):
    """RUL en ciclos para un arreglo (n, W, F)."""
    model.eval()                   # apaga el dropout; BatchNorm usa sus promedios de train
    out = []
    with torch.no_grad():          # sin gradientes: solo inferencia
        for i in range(0, len(X), 1024):
            out.append(model(torch.from_numpy(X[i:i + 1024]).to(device)).cpu())
    return torch.cat(out).numpy() * RUL_MAX


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def phm_score(y, p):
    """Score asimétrico PHM08. d > 0 es predicción tardía (dice que le queda más vida
    de la real) y se castiga más (divisor 10) que una temprana (divisor 13).
    Es una SUMA: solo sirve para comparar modelos sobre esta misma validación
    (7539 ventanas), no con los scores de las 218 unidades de test."""
    d = p.astype(np.float64) - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ============ 4) ENTRENAMIENTO ============
# Referencias en esta misma validación: predecir la media = 43.4, regresión lineal = 23.7
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
best_rmse, best_state, wait = np.inf, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()                              # activa dropout; BatchNorm usa el batch actual
    total = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        y_hat = model(xb)                      # 1) predicción con los pesos actuales
        loss = loss_fn(y_hat, yb)              # 2) error cuadrático medio del batch
        opt.zero_grad()                        # 3) borra gradientes (PyTorch los acumula)
        loss.backward()                        # 4) d(loss)/d(peso) para cada peso
        opt.step()                             # 5) Adam actualiza todos los pesos
        total += loss.item() * len(xb)

    # RMSE de train en ciclos: promedio durante la época, con dropout activo
    train_rmse = float(np.sqrt(total / N)) * RUL_MAX
    val_rmse = rmse(y_val, predict(X_val))
    mark = ""
    if val_rmse < best_rmse:                   # guarda el mejor modelo según validación
        best_rmse, wait, mark = val_rmse, 0, " *"
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, HERE / "cnn_best.pt")
    else:
        wait += 1
    print(f"época {epoch:3d} | RMSE train {train_rmse:6.2f} | RMSE val {val_rmse:6.2f}{mark}")
    if wait >= PATIENCE:
        print(f"Early stopping: {PATIENCE} épocas sin mejorar en validación")
        break

# ============ 5) EVALUACIÓN Y TEST ============
model.load_state_dict(best_state)
p_val = predict(X_val)
print(f"\nMejor modelo | RMSE val {rmse(y_val, p_val):.2f} | score PHM08 val {phm_score(y_val, p_val):.0f}")

p_test = predict(X_test)                       # 218 motores: última ventana de cada uno
np.savetxt(HERE / "pred_test_cnn.txt", p_test, fmt="%.2f")
print("Guardado: cnn_best.pt y pred_test_cnn.txt")
