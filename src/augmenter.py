import os
import random
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

# ------------------------
# Paths (raw strings to avoid \S warnings)
# ------------------------
source_root = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\dataset_raw"
target_root = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\dataset_final"

# Train/Val split ratio
train_split = 0.8
aug_factor = 3  # number of augmentations per image

# ------------------------
# Augmentations
# ------------------------
def augment_image(img):
    """Apply simple augmentations to increase dataset variety."""
    augmentations = []

    # Flip
    augmentations.append(ImageOps.mirror(img))
    augmentations.append(ImageOps.flip(img))

    # Rotate
    augmentations.append(img.rotate(random.randint(-25, 25)))

    # Color/contrast
    augmentations.append(ImageEnhance.Contrast(img).enhance(1.5))
    augmentations.append(ImageEnhance.Color(img).enhance(0.7))

    return augmentations

# ------------------------
# Processing logic
# ------------------------
summary = {}

def save_images(file_list, split, class_name):
    out_dir = os.path.join(target_root, split, class_name)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for f in tqdm(file_list, desc=f"{class_name}-{split}"):
        try:
            img = Image.open(f).convert("RGBA").convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping bad file: {f} ({e})")
            continue

        base = os.path.splitext(os.path.basename(f))[0]

        # Save original
        img.save(os.path.join(out_dir, f"{base}.jpg"))
        count += 1

        # Save augmentations
        for i, aug in enumerate(augment_image(img)[:aug_factor]):
            aug.save(os.path.join(out_dir, f"{base}_aug{i}.jpg"))
            count += 1

    summary[f"{class_name}-{split}"] = count


def process_class(class_path, class_name):
    files = []
    for root, _, filenames in os.walk(class_path):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                files.append(os.path.join(root, fn))

    print(f"📂 Found {len(files)} images for class '{class_name}' in {class_path}")

    random.shuffle(files)
    split_idx = int(len(files) * train_split)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    save_images(train_files, "train", class_name)
    save_images(val_files, "val", class_name)


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    classes = ["graphs", "random"]  # add more if you expand
    for cls in classes:
        process_class(os.path.join(source_root, cls), cls)

    print("\n✅ Dataset ready!\n")
    for k, v in summary.items():
        print(f"{k}: {v} images")
