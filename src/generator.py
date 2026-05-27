import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

# -------------------------------
# Settings
# -------------------------------
output_dir = "humanized_linear_dataset"
os.makedirs(output_dir, exist_ok=True)

num_samples = 2000
img_size = (224, 224)
csv_file = os.path.join(output_dir, "labels.csv")


# -------------------------------
# Helper Functions
# -------------------------------

def generate_humanized_linear_graph(m=None, c=None, length=50, jitter=0.02):
    """
    Generates a "humanized" linear graph with slight irregularities
    """

    if m is None:
        m = random.uniform(-5, 5)
    if c is None:
        c = random.uniform(-1, 1)

    # Non-uniform x values to simulate human plotting errors
    x = np.linspace(0, 1, length)
    x += np.random.normal(0, jitter / 2, size=x.shape)

    # y-values with noise
    y = m * x + c + np.random.normal(0, jitter, size=x.shape)

    plt.figure(figsize=(2.5, 2.5))

    # Hand-drawn style: plot points as scatter or thin line with wobbles
    for i in range(len(x) - 1):
        x_vals = [x[i], x[i + 1]]
        y_vals = [y[i], y[i + 1]]
        # Add small random wobble
        x_vals = [v + random.uniform(-0.01, 0.01) for v in x_vals]
        y_vals = [v + random.uniform(-0.01, 0.01) for v in y_vals]
        plt.plot(x_vals, y_vals, color='black', linewidth=random.uniform(1, 2))

    # Scatter to simulate plotted dots
    plt.scatter(x, y, color='black', s=random.uniform(5, 10))

    # Axes with numbers
    plt.xlim(0, 1)
    plt.ylim(-6, 6)
    plt.xticks(np.linspace(0, 1, 5))
    plt.yticks(np.linspace(-6, 6, 7))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    # Save to temporary file
    tmp_file = "temp.png"
    plt.savefig(tmp_file, dpi=100)
    plt.close()

    # Open image and convert to RGB
    img = Image.open(tmp_file).convert("RGB")
    img = img.resize(img_size)

    # Augment image slightly
    img = augment_image(img)

    return img, m, c


def augment_image(img):
    # Random rotation
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-10, 10))
    # Random horizontal flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    # Gaussian blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


# -------------------------------
# Generate Dataset
# -------------------------------
labels = []

print(f"Generating {num_samples} humanized linear graphs...")
for i in tqdm(range(num_samples)):
    img, m, c = generate_humanized_linear_graph()
    filename = f"graph_{i:04d}.png"
    img.save(os.path.join(output_dir, filename))
    labels.append([filename, m, c])

# -------------------------------
# Save CSV Labels
# -------------------------------
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "m", "c"])
    writer.writerows(labels)

print(f"✅ Dataset generated in '{output_dir}' with labels saved to '{csv_file}'")
