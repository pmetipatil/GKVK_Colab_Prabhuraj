"""
Neural Network from scratch — Farming Yield Classifier
Two features: Soil Moisture (%) and Sunlight Hours/day
Also demonstrates WHY CNN is needed for spatial/image farm data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# PART 1 — DATASET
# ══════════════════════════════════════════════════════════════
# 60 farm plots, 2 features: soil moisture (%), sunlight hrs/day
# High yield = moisture > 55% AND sunlight > 5.5 hrs
n = 60
moisture  = np.random.uniform(20, 90, n)   # x1
sunlight  = np.random.uniform(2,  10, n)   # x2
X_raw = np.column_stack([moisture, sunlight])

# Labels: high yield (1) if both features in favourable range
y_raw = ((moisture > 55) & (sunlight > 5.5)).astype(float).reshape(-1, 1)

# Normalise to [0,1]
X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)
y = y_raw

# Train/test split
idx = np.random.permutation(n)
split = int(0.8 * n)
X_train, y_train = X[idx[:split]], y[idx[:split]]
X_test,  y_test  = X[idx[split:]], y[idx[split:]]

# ══════════════════════════════════════════════════════════════
# PART 2 — NEURAL NETWORK (all from scratch, no sklearn)
# ══════════════════════════════════════════════════════════════

def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(float)
def sigmoid(z):     return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def bce_loss(y_true, y_pred):
    eps = 1e-12
    return -np.mean(y_true * np.log(y_pred + eps) +
                    (1 - y_true) * np.log(1 - y_pred + eps))

class FarmNN:
    """
    Architecture: Input(2) → Hidden1(6) → Hidden2(4) → Output(1)
    Activations : ReLU / ReLU / Sigmoid
    Optimizer   : Mini-batch SGD with momentum
    """
    def __init__(self, lr=0.05, momentum=0.9):
        self.lr = lr
        self.mu = momentum
        # Xavier init
        self.W1 = np.random.randn(2, 6) * np.sqrt(2/2)
        self.b1 = np.zeros((1, 6))
        self.W2 = np.random.randn(6, 4) * np.sqrt(2/6)
        self.b2 = np.zeros((1, 4))
        self.W3 = np.random.randn(4, 1) * np.sqrt(2/4)
        self.b3 = np.zeros((1, 1))
        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, X):
        self.X  = X
        self.z1 = X  @ self.W1 + self.b1;   self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3; self.a3 = sigmoid(self.z3)
        return self.a3

    def backward(self, y):
        m = y.shape[0]
        # Output layer gradient
        dL_da3 = (self.a3 - y) / m
        dL_dz3 = dL_da3 * self.a3 * (1 - self.a3)   # sigmoid derivative
        dL_dW3 = self.a2.T @ dL_dz3
        dL_db3 = dL_dz3.sum(axis=0, keepdims=True)
        # Hidden layer 2
        dL_da2 = dL_dz3 @ self.W3.T
        dL_dz2 = dL_da2 * relu_grad(self.z2)
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = dL_dz2.sum(axis=0, keepdims=True)
        # Hidden layer 1
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * relu_grad(self.z1)
        dL_dW1 = self.X.T @ dL_dz1
        dL_db1 = dL_dz1.sum(axis=0, keepdims=True)
        # SGD + momentum update
        for (W, b, gW, gb, vW, vb) in [
            (self.W1, self.b1, dL_dW1, dL_db1, self.vW1, self.vb1),
            (self.W2, self.b2, dL_dW2, dL_db2, self.vW2, self.vb2),
            (self.W3, self.b3, dL_dW3, dL_db3, self.vW3, self.vb3),
        ]:
            vW[:] = self.mu * vW - self.lr * gW
            vb[:] = self.mu * vb - self.lr * gb
            W += vW
            b += vb

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

# ── Train ──
model = FarmNN(lr=0.05, momentum=0.9)
epochs        = 600
batch_size    = 16
train_losses, test_losses   = [], []
train_accs,   test_accs     = [], []

for epoch in range(epochs):
    # Mini-batch loop
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        Xb = X_train[perm[i:i+batch_size]]
        yb = y_train[perm[i:i+batch_size]]
        model.forward(Xb)
        model.backward(yb)
    # Record metrics every 5 epochs
    if epoch % 5 == 0:
        tr_loss = bce_loss(y_train, model.forward(X_train))
        te_loss = bce_loss(y_test,  model.forward(X_test))
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(model.accuracy(X_train, y_train))
        test_accs.append(model.accuracy(X_test,  y_test))

final_train_acc = model.accuracy(X_train, y_train) * 100
final_test_acc  = model.accuracy(X_test,  y_test)  * 100

# ── Metrics ──
y_pred_test = model.predict(X_test).flatten()
y_true_test = y_test.flatten().astype(int)
TP = ((y_pred_test==1) & (y_true_test==1)).sum()
TN = ((y_pred_test==0) & (y_true_test==0)).sum()
FP = ((y_pred_test==1) & (y_true_test==0)).sum()
FN = ((y_pred_test==0) & (y_true_test==1)).sum()
precision = TP / (TP + FP + 1e-9)
recall    = TP / (TP + FN + 1e-9)
f1        = 2 * precision * recall / (precision + recall + 1e-9)

print(f"Train Accuracy : {final_train_acc:.1f}%")
print(f"Test  Accuracy : {final_test_acc:.1f}%")
print(f"Precision      : {precision:.3f}")
print(f"Recall         : {recall:.3f}")
print(f"F1 Score       : {f1:.3f}")
print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")

# ══════════════════════════════════════════════════════════════
# PART 3 — SPATIAL FIELD GRID (why CNN is needed)
# ══════════════════════════════════════════════════════════════
# Imagine the farm field as a 10×10 grid of plots.
# Each cell's colour = soil moisture reading.
# Diseased patches appear as contiguous low-moisture blobs.
# A plain NN treats each pixel independently — it CANNOT detect
# the spatial neighbourhood pattern.  CNN's convolution window
# slides across and detects "low-moisture surrounded by low-moisture"
# = likely diseased patch.

field = np.random.uniform(50, 90, (10, 10))  # healthy baseline
# Inject 3 diseased blobs (low moisture)
blobs = [(2,2,3,3),(6,1,2,2),(3,7,3,2)]
for r,c,h,w in blobs:
    field[r:r+h, c:c+w] = np.random.uniform(15, 30, (h, w))

# What a 3×3 average-pooling filter "sees"
kernel = np.ones((3,3)) / 9.0
conv_out = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        conv_out[i,j] = (field[i:i+3, j:j+3] * kernel).sum()

# ══════════════════════════════════════════════════════════════
# PART 4 — FULL FIGURE
# ══════════════════════════════════════════════════════════════
COLORS  = ["#378ADD","#1D9E75","#D85A30","#534AB7","#BA7517"]
BG      = "#F7F6F2"
fig = plt.figure(figsize=(18, 16), facecolor=BG)
fig.suptitle("Neural Network & CNN — Farm Yield Classification",
             fontsize=18, fontweight="bold", color="#2C2C2A", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36)

# ── 1. Decision boundary ──
ax1 = fig.add_subplot(gs[0, 0])
h = 0.01
xx, yy = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax1.contourf(xx, yy, Z, alpha=0.25, cmap="RdYlGn", levels=[-0.5,0.5,1.5])
ax1.contour(xx, yy, Z, levels=[0.5], colors="#2C2C2A", linewidths=1.2, linestyles="--")
for label, color, marker, name in [(0,"#D85A30","o","Low yield"), (1,"#1D9E75","^","High yield")]:
    m = y_train.flatten() == label
    ax1.scatter(X_train[m,0], X_train[m,1], c=color, marker=marker,
                s=60, edgecolors="white", linewidths=0.6, label=name, zorder=3)
ax1.set_xlabel("Soil moisture (normalised)", fontsize=9)
ax1.set_ylabel("Sunlight hrs (normalised)", fontsize=9)
ax1.set_title("Decision boundary", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_facecolor("#F0EEE8"); ax1.grid(True, color="white", linewidth=0.8)

# ── 2. Loss curve ──
ax2 = fig.add_subplot(gs[0, 1])
ep_axis = np.arange(0, epochs, 5)
ax2.plot(ep_axis, train_losses, color=COLORS[0], linewidth=2, label="Train loss")
ax2.plot(ep_axis, test_losses,  color=COLORS[2], linewidth=2, linestyle="--", label="Test loss")
ax2.fill_between(ep_axis, train_losses, alpha=0.12, color=COLORS[0])
ax2.set_xlabel("Epoch", fontsize=9); ax2.set_ylabel("BCE Loss", fontsize=9)
ax2.set_title("Training & test loss", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8); ax2.set_facecolor("#F0EEE8"); ax2.grid(True, color="white", linewidth=0.8)

# ── 3. Accuracy curve ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(ep_axis, [a*100 for a in train_accs], color=COLORS[0], linewidth=2, label="Train acc")
ax3.plot(ep_axis, [a*100 for a in test_accs],  color=COLORS[1], linewidth=2, linestyle="--", label="Test acc")
ax3.axhline(100, color="#D3D1C7", linestyle=":", linewidth=1)
ax3.set_xlabel("Epoch", fontsize=9); ax3.set_ylabel("Accuracy (%)", fontsize=9)
ax3.set_title("Accuracy over training", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8); ax3.set_facecolor("#F0EEE8"); ax3.grid(True, color="white", linewidth=0.8)

# ── 4. Confusion matrix ──
ax4 = fig.add_subplot(gs[1, 0])
cm = np.array([[TN, FP],[FN, TP]])
im = ax4.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max()+1)
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(["Pred Low","Pred High"], fontsize=9)
ax4.set_yticklabels(["True Low","True High"], fontsize=9)
ax4.set_title("Confusion matrix (test)", fontsize=10, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i,j]), ha="center", va="center",
                 fontsize=18, fontweight="bold",
                 color="white" if cm[i,j] > cm.max()/2 else "#2C2C2A")
ax4.set_facecolor("#F0EEE8")

# ── 5. Evaluation metrics bar ──
ax5 = fig.add_subplot(gs[1, 1])
m_names  = ["Train\nAcc", "Test\nAcc", "Precision", "Recall", "F1\nScore"]
m_values = [final_train_acc/100, final_test_acc/100, precision, recall, f1]
bars = ax5.bar(m_names, m_values, color=COLORS, edgecolor="white", linewidth=0.8, width=0.5)
for bar, val in zip(bars, m_values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2C2C2A")
ax5.set_ylim(0, 1.15); ax5.set_ylabel("Score", fontsize=9)
ax5.set_title("Evaluation metrics", fontsize=10, fontweight="bold")
ax5.set_facecolor("#F0EEE8"); ax5.grid(True, color="white", linewidth=0.8, axis="y")

# ── 6. Weight gradient heatmap (W1 layer) ──
ax6 = fig.add_subplot(gs[1, 2])
im2 = ax6.imshow(model.W1, cmap="RdBu", aspect="auto", vmin=-2, vmax=2)
ax6.set_xlabel("Hidden neurons (layer 1)", fontsize=9)
ax6.set_ylabel("Input features", fontsize=9)
ax6.set_yticks([0,1]); ax6.set_yticklabels(["Moisture","Sunlight"], fontsize=9)
ax6.set_title("Learned weights — input→hidden1", fontsize=10, fontweight="bold")
plt.colorbar(im2, ax=ax6, fraction=0.04, pad=0.04)
ax6.set_facecolor("#F0EEE8")

# ── 7. Farm field moisture heatmap ──
ax7 = fig.add_subplot(gs[2, 0])
cmap7 = plt.cm.RdYlGn
im7 = ax7.imshow(field, cmap=cmap7, vmin=10, vmax=90, interpolation="nearest", aspect="auto")
ax7.set_title("Farm field: soil moisture grid", fontsize=10, fontweight="bold")
ax7.set_xlabel("Column", fontsize=9); ax7.set_ylabel("Row", fontsize=9)
plt.colorbar(im7, ax=ax7, fraction=0.04, pad=0.04, label="Moisture %")
# Mark disease blobs
for r,c,h,w in blobs:
    rect = plt.Rectangle((c-0.5,r-0.5), w, h, fill=False,
                          edgecolor="white", linewidth=2, linestyle="--")
    ax7.add_patch(rect)
ax7.text(7.5, 0.5, "diseased\nzones", fontsize=8, color="white", fontweight="bold",
         ha="center", va="top")
ax7.set_facecolor("#F0EEE8")

# ── 8. Convolution output ──
ax8 = fig.add_subplot(gs[2, 1])
im8 = ax8.imshow(conv_out, cmap=cmap7, vmin=10, vmax=90, interpolation="nearest", aspect="auto")
ax8.set_title("After 3×3 mean filter (CNN conv)", fontsize=10, fontweight="bold")
ax8.set_xlabel("Column", fontsize=9); ax8.set_ylabel("Row", fontsize=9)
plt.colorbar(im8, ax=ax8, fraction=0.04, pad=0.04, label="Avg moisture")
ax8.set_facecolor("#F0EEE8")
ax8.text(4, 8.5, "Low-moisture patches now spatially highlighted",
         fontsize=8, ha="center", color="#2C2C2A",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D3D1C7", lw=0.5))

# ── 9. NN vs CNN comparison panel ──
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
rows = [
    ["Feature",                  "Plain NN",              "CNN"],
    ["Input",                    "Flat feature vector",   "Raw grid/image"],
    ["Spatial awareness",        "None — pixels treated\nindependently",
                                                          "Yes — conv kernel\nscans neighbourhoods"],
    ["Parameter sharing",        "All weights unique",    "Kernel weights shared\nacross all positions"],
    ["Disease patch detection",  "Cannot detect\nblob shape",
                                                          "Detects contiguous\nlow-moisture blobs"],
    ["When to use",              "Tabular features\n(2-feature farm data)",
                                                          "Grid/image data\n(field maps, drone imagery)"],
]
col_widths = [0.28, 0.36, 0.36]
table = ax9.table(cellText=rows[1:], colLabels=rows[0],
                  cellLoc="left", loc="center",
                  colWidths=col_widths)
table.auto_set_font_size(False); table.set_fontsize(8.0); table.scale(1, 2.0)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#D3D1C7")
    if r == 0:
        cell.set_facecolor("#2C2C2A"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#FAECE7" if r % 2 == 0 else "#FEF6F3")
    elif c == 2:
        cell.set_facecolor("#E1F5EE" if r % 2 == 0 else "#F0FAF6")
    else:
        cell.set_facecolor("#F7F6F2" if r % 2 == 0 else "white")
ax9.set_title("Plain NN vs CNN — key differences", fontsize=10, fontweight="bold", pad=8)

plt.savefig("/mnt/user-data/outputs/nn_farming.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved.")
