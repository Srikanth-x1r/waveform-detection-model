import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
train_dir = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L3train"
val_dir = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L3val"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

classes = [
    "linear", "quadratic", "exponential", "sinusoidal", "square",
    "triangular", "sawtooth", "sinc", "step_ramp", "gaussian", "piecewise"
]

n_train = 1000
n_val = 250

# CSV label files
train_labels = []
val_labels = []


# ---------------------------
# SIGNAL GENERATORS
# ---------------------------
def generate_signal(cls, x):
    if cls == "linear":
        m = np.random.uniform(-5, 5)
        c = np.random.uniform(-3, 3)
        return m * x + c

    elif cls == "quadratic":
        a = np.random.uniform(-2, 2)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-3, 3)
        return a * x ** 2 + b * x + c

    elif cls == "exponential":
        a = np.random.uniform(-2, 2)
        return np.exp(a * x / 5)

    elif cls == "sinusoidal":
        A = np.random.uniform(0.5, 2)
        f = np.random.uniform(0.5, 3)
        p = np.random.uniform(0, np.pi)
        return A * np.sin(2 * np.pi * f * x + p)

    elif cls == "square":
        return np.sign(np.sin(2 * np.pi * 2 * x))

    elif cls == "triangular":
        return 2 * np.abs(2 * (x - np.floor(x + 0.5))) - 1

    elif cls == "sawtooth":
        return 2 * (x / np.max(x) - np.floor(0.5 + x / np.max(x)))

    elif cls == "sinc":
        return np.sinc(x)

    elif cls == "step_ramp":
        return np.where(x < 0, 0, x)  # ramp starting at 0

    elif cls == "gaussian":
        mu = np.random.uniform(-1, 1)
        sigma = np.random.uniform(0.2, 1.0)
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    elif cls == "piecewise":
        y = np.piecewise(x, [x < -1, (x >= -1) & (x < 1), x >= 1],
                         [lambda x: -1 * x, lambda x: x ** 2, lambda x: np.sin(x)])
        return y


# ---------------------------
# IMAGE GENERATOR
# ---------------------------
def save_plot(cls, idx, dataset_type):
    x = np.linspace(-5, 5, 400)
    y = generate_signal(cls, x)

    plt.figure(figsize=(3, 3))
    plt.plot(x, y, color="blue")
    plt.grid(True, linewidth=0.3)
    plt.axis("on")

    fname = f"{cls}_{idx}.png"
    if dataset_type == "train":
        path = os.path.join(train_dir, fname)
        train_labels.append([fname, classes.index(cls)])
    else:
        path = os.path.join(val_dir, fname)
        val_labels.append([fname, classes.index(cls)])

    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------
# GENERATE DATASET
# ---------------------------
for cls in classes:
    for i in range(n_train):
        save_plot(cls, i, "train")
    for i in range(n_val):
        save_plot(cls, i, "val")

# ---------------------------
# SAVE LABELS
# ---------------------------
pd.DataFrame(train_labels, columns=["filename", "label"]).to_csv("S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L3train/train_labels.csv", index=False)
pd.DataFrame(val_labels, columns=["filename", "label"]).to_csv(r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L3val\val_labels.csv", index=False)

print("✅ Dataset generated: 11 classes, train + val")
