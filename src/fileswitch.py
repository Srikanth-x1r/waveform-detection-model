import os, shutil

base_dir = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\dataset_final"

for split in ["train", "val"]:
    src_graphs = os.path.join(base_dir, split, "graphs")
    src_graph = os.path.join(base_dir, split, "graph")
    dst = os.path.join(base_dir, split, "graph")

    os.makedirs(dst, exist_ok=True)

    def move_with_unique(src):
        if not os.path.exists(src):
            return
        for f in os.listdir(src):
            src_path = os.path.join(src, f)
            if not os.path.isfile(src_path):
                continue
            # Ensure unique filename
            base, ext = os.path.splitext(f)
            new_name = f
            counter = 1
            while os.path.exists(os.path.join(dst, new_name)):
                new_name = f"{base}_{counter}{ext}"
                counter += 1
            shutil.move(src_path, os.path.join(dst, new_name))
        os.rmdir(src)

    # Merge both folders into dst
    move_with_unique(src_graphs)
    move_with_unique(src_graph)

# ✅ Final sanity check
for split in ["train", "val"]:
    for cls in ["graph", "random"]:
        folder = os.path.join(base_dir, split, cls)
        print(f"{split}/{cls}: {len(os.listdir(folder))} images")
