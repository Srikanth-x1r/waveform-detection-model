import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------
# CONFIG
# --------------------
TRAIN_DIR = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L3train"
VAL_DIR = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\L3val"
TRAIN_CSV = os.path.join(TRAIN_DIR, "train_labels.csv")
VAL_CSV = os.path.join(VAL_DIR, "val_labels.csv")

PRETRAINED_PATH = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\slope_intercept_model_final.pth"
MODEL_PATH = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\functions_classifier.pth"

BATCH_SIZE = 32
EPOCHS = 7
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 11

idx_to_class = {
    0: "linear", 1: "quadratic", 2: "exponential", 3: "sinusoidal", 4: "square",
    5: "triangular", 6: "sawtooth", 7: "sinc", 8: "step_ramp", 9: "gaussian", 10: "piecewise"
}

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
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------
# TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = GraphDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)
val_dataset = GraphDataset(VAL_CSV, VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------
# MODEL (LOAD PRETRAINED BACKBONE)
# --------------------
# Create ResNet18 architecture
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # dummy for loading pretrained weights

# Load pretrained weights
state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
model.load_state_dict(filtered_dict, strict=False)

# Replace fc for 11-class classifier
model.fc = nn.Linear(num_features, NUM_CLASSES)
model = model.to(DEVICE)
print("✅ Pretrained backbone loaded. Final classifier replaced for 11 classes.")

# --------------------
# TRAINING SETUP
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------
# TRAIN FUNCTION
# --------------------
def train_model(model, train_loader, val_loader, epochs):
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {train_loss:.4f} "
              f"Accuracy: {train_acc:.2f}%")

    # --------------------
    # VALIDATION
    # --------------------
    model.eval()
    val_preds = []
    val_truths = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_truths.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_truths, val_preds) * 100
    print("\n✅ Final Validation Accuracy: {:.2f}%".format(val_acc))
    print("\nClassification Report:\n", classification_report(val_truths, val_preds, target_names=list(idx_to_class.values())))
    print("\nConfusion Matrix:\n", confusion_matrix(val_truths, val_preds))

    elapsed_time = time.time() - start_time
    print(f"\n⏱ Training + Evaluation Time: {elapsed_time/60:.2f} minutes")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

# --------------------
# RUN TRAINING
# --------------------
train_model(model, train_loader, val_loader, EPOCHS)
