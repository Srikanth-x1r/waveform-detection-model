import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --------------------
# CONFIG
# --------------------
OUTPUT_DIR = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\humanized_linear_dataset1"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

SAMPLES_PER_TYPE = 2000  # 2000 images per quadrant type
IMG_SIZE = (224, 224)
X_RANGE = (-10, 10)
Y_RANGE = (-50, 50)


# --------------------
# HUMANIZED LINE
# --------------------
def humanized_line(x, m, c):
    y = m * x + c
    noise = np.random.normal(0, 0.5, size=y.shape)  # small jitter
    return y + noise


# --------------------
# GENERATE DATASET
# --------------------
def generate_quadrant_dataset(num_samples, quadrant_type, output_dir, start_idx=0):
    records = []
    for i in range(num_samples):
        # Set slope and intercept ranges based on quadrant type
        if quadrant_type == "Q1":  # +slope, +intercept
            m = np.random.uniform(0.5, 5.0)
            c = np.random.uniform(0.0, 10.0)
        elif quadrant_type == "Q2":  # -slope, +intercept
            m = np.random.uniform(-5.0, -0.5)
            c = np.random.uniform(0.0, 10.0)
        elif quadrant_type == "Q3":  # -slope, -intercept
            m = np.random.uniform(-5.0, -0.5)
            c = np.random.uniform(-10.0, 0.0)
        elif quadrant_type == "Q4":  # +slope, -intercept
            m = np.random.uniform(0.5, 5.0)
            c = np.random.uniform(-10.0, 0.0)
        else:
            raise ValueError("Unknown quadrant type")

        x = np.linspace(*X_RANGE, 100)
        y = humanized_line(x, m, c)

        # Plot line
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, color='blue', linewidth=np.random.uniform(1, 3))
        plt.xlim(*X_RANGE)
        plt.ylim(*Y_RANGE)
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")

        filename = f"{quadrant_type}_{start_idx + i}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close()

        records.append([filename, m, c])

    return records


# --------------------
# CREATE TRAIN AND VAL SET
# --------------------
def create_full_dataset():
    train_records = []
    val_records = []

    quadrant_types = ["Q1", "Q2", "Q3", "Q4"]
    for quadrant in quadrant_types:
        # 2000 train images per quadrant
        train_records.extend(
            generate_quadrant_dataset(SAMPLES_PER_TYPE, quadrant, TRAIN_DIR, start_idx=len(train_records)))
        # 250 val images per quadrant
        val_records.extend(generate_quadrant_dataset(int(SAMPLES_PER_TYPE * 0.125), quadrant, VAL_DIR,
                                                     start_idx=len(val_records)))  # 250*4=1000

    # Save CSVs
    pd.DataFrame(train_records, columns=["filename", "m", "c"]).to_csv(os.path.join(TRAIN_DIR, "labels.csv"),
                                                                       index=False)
    pd.DataFrame(val_records, columns=["filename", "m", "c"]).to_csv(os.path.join(VAL_DIR, "labels_val.csv"),
                                                                     index=False)
    print(f"✅ Dataset created! Train: {len(train_records)} images, Val: {len(val_records)} images")


# --------------------
# RUN GENERATION
# --------------------
create_full_dataset()
