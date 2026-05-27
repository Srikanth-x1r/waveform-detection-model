import os
import random
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from bing_image_downloader import downloader

# -------------------------------
# Step 1: Helper functions
def augment_image(img):
    img = img.convert('RGB')
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-10, 10))
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.autocontrast(img)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    if random.random() < 0.5:
        arr = np.array(img)
        noise = np.random.normal(0, 15, arr.shape).astype(np.uint8)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr)
    img = img.resize((224, 224))
    return img

def save_augmented_images(source_dir, target_dir, class_name, val_ratio=0.2):
    img_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    random.shuffle(img_files)
    split_idx = int(len(img_files) * (1 - val_ratio))

    train_dir = os.path.join(target_dir, 'train', class_name)
    val_dir = os.path.join(target_dir, 'val', class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for i, f in enumerate(tqdm(img_files, desc=f"Augmenting {class_name}")):
        try:
            img_path = os.path.join(source_dir, f)
            img = Image.open(img_path)
            aug_img = augment_image(img)
            save_path = os.path.join(train_dir if i < split_idx else val_dir, f)
            aug_img.save(save_path)
        except:
            continue

# -------------------------------
# Step 2: Download images from Bing

def download_bing_images(queries, limit, save_root):
    for q in queries:
        out_dir = os.path.join(save_root, q.replace(' ', '_'))
        os.makedirs(out_dir, exist_ok=True)
        print(f"Downloading '{q}' images...")
        downloader.download(
            q,
            limit=limit,
            output_dir=save_root,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )

# -------------------------------
# Step 3: Main script

dataset_raw_dir = "dataset_raw"
dataset_final_dir = "dataset_final"

# Graph images (~1000)
graph_queries = [
    "line graph",
    "scatter plot",
    "bar chart",
    "paper graph",
    "digital signal waveform",
    "discrete signal plot",
    "scientific plot"
]
download_bing_images(graph_queries, limit=150, save_root=os.path.join(dataset_raw_dir, "graphs"))

# Random images (~1000)
random_queries = ["nature photo", "animals", "cars", "buildings", "people"]
download_bing_images(random_queries, limit=200, save_root=os.path.join(dataset_raw_dir, "random"))

# Augment and split
save_augmented_images(os.path.join(dataset_raw_dir, "graphs"), dataset_final_dir, "graph")
save_augmented_images(os.path.join(dataset_raw_dir, "random"), dataset_final_dir, "no_graph")

print("Dataset ready at 'dataset_final/train' and 'dataset_final/val'")
