import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --------------------
# CONFIG
# --------------------
TRAIN_DIR = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\humanized_linear_dataset1\train"
VAL_DIR   = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\humanized_linear_dataset1\val"
TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels.csv")
VAL_LABELS   = os.path.join(VAL_DIR, "labels_val.csv")

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\slope_intercept_model_final.pth"

# --------------------
# DATASET CLASS
# --------------------
class GraphDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        labels = self.data.iloc[idx, 1:].values.astype("float32")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels, dtype=torch.float32)

# --------------------
# TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = GraphDataset(TRAIN_LABELS, TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = GraphDataset(VAL_LABELS, VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------
# MODEL
# --------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # slope (m) and intercept (c)
model = model.to(DEVICE)

# --------------------
# LOSS & OPTIMIZER
# --------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------
# TRAINING LOOP
# --------------------
def train(model, train_loader, val_loader, epochs):
    best_rmse = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        # Train metrics
        preds = np.vstack(all_preds)
        truths = np.vstack(all_labels)
        rmse = np.sqrt(((preds - truths) ** 2).mean())
        tolerance = 0.2
        accuracy = (np.all(np.abs(preds - truths) <= tolerance, axis=1)).mean() * 100

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader):.4f} "
              f"RMSE: {rmse:.4f} Accuracy: {accuracy:.2f}%")

    # --------------------
    # EVALUATION
    # --------------------
    model.eval()
    val_preds, val_truths, val_filenames = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_preds.append(outputs.cpu().numpy())
            val_truths.append(labels.cpu().numpy())

    val_preds = np.vstack(val_preds)
    val_truths = np.vstack(val_truths)

    val_rmse = np.sqrt(((val_preds - val_truths) ** 2).mean())
    val_accuracy = (np.all(np.abs(val_preds - val_truths) <= tolerance, axis=1)).mean() * 100
    print(f"\n✅ Validation Results - RMSE: {val_rmse:.4f} Accuracy: {val_accuracy:.2f}%")

    # Print sample predictions
    print("\nSample predictions vs actual:")
    for i in range(10):
        print(f"Actual m: {val_truths[i,0]:.4f}, Predicted m: {val_preds[i,0]:.4f} | "
              f"Actual c: {val_truths[i,1]:.4f}, Predicted c: {val_preds[i,1]:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved as {MODEL_PATH}")

# --------------------
# RUN TRAINING + EVALUATION
# --------------------
train(model, train_loader, val_loader, EPOCHS)
