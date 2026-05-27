import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
TRAIN_DIR = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L4train"
VAL_DIR   = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L4val"
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

classes = [
    "linear", "quadratic", "exponential", "sinusoidal", "square",
    "triangular", "sawtooth", "sinc", "step_ramp", "gaussian", "piecewise"
]

n_train = 1000
n_val   = 250

# CSV label files
train_labels = []
val_labels   = []

# ---------------------------
# SIGNAL GENERATORS
# ---------------------------
def generate_signal(cls, x):
    params = {}
    if cls == "linear":
        m = np.random.uniform(-5, 5)
        c = np.random.uniform(-3, 3)
        y = m*x + c
        params = {"m": m, "c": c}

    elif cls == "quadratic":
        a = np.random.uniform(-2, 2)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-3, 3)
        y = a*x**2 + b*x + c
        params = {"a": a, "b": b, "c": c}

    elif cls == "exponential":
        a = np.random.uniform(-2, 2)
        y = np.exp(a*x/5)
        params = {"a": a}

    elif cls == "sinusoidal":
        A = np.random.uniform(0.5, 2)
        f = np.random.uniform(0.5, 3)
        p = np.random.uniform(0, np.pi)
        y = A * np.sin(2*np.pi*f*x + p)
        params = {"A": A, "f": f, "p": p}

    elif cls == "square":
        y = np.sign(np.sin(2*np.pi*2*x))
        params = {}

    elif cls == "triangular":
        y = 2*np.abs(2*(x-np.floor(x+0.5)))-1
        params = {}

    elif cls == "sawtooth":
        y = 2*(x/np.max(x)-np.floor(0.5 + x/np.max(x)))
        params = {}

    elif cls == "sinc":
        y = np.sinc(x)
        params = {}

    elif cls == "step_ramp":
        y = np.where(x<0, 0, x)
        params = {}

    elif cls == "gaussian":
        mu = np.random.uniform(-1,1)
        sigma = np.random.uniform(0.2,1.0)
        y = np.exp(-(x-mu)**2 / (2*sigma**2))
        params = {"mu": mu, "sigma": sigma}

    elif cls == "piecewise":
        y = np.piecewise(x, [x<-1, (x>=-1) & (x<1), x>=1],
                         [lambda x: -1*x, lambda x: x**2, lambda x: np.sin(x)])
        params = {}

    return y, params

# ---------------------------
# IMAGE GENERATOR
# ---------------------------
def save_plot(cls, idx, dataset_type):
    x = np.linspace(-5,5,400)
    y, params = generate_signal(cls,x)

    plt.figure(figsize=(3,3))
    plt.plot(x,y,color="blue")
    plt.grid(True, linewidth=0.3)
    plt.axis("on")

    fname = f"{cls}_{idx}.png"
    if dataset_type=="train":
        path = os.path.join(TRAIN_DIR,fname)
        row = {"filename": fname, "label": classes.index(cls), **params}
        train_labels.append(row)
    else:
        path = os.path.join(VAL_DIR,fname)
        row = {"filename": fname, "label": classes.index(cls), **params}
        val_labels.append(row)

    plt.savefig(path,dpi=100,bbox_inches="tight")
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
pd.DataFrame(train_labels).to_csv(os.path.join(TRAIN_DIR,"train_labels.csv"), index=False)
pd.DataFrame(val_labels).to_csv(os.path.join(VAL_DIR,"val_labels.csv"), index=False)

print("✅ Dataset generated with parameters + labels")
