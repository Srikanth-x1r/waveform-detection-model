import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# -------------------------
# Paths
# -------------------------
dataset_dir = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\dataset_final"

# -------------------------
# Transforms
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Main training function
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Datasets & DataLoaders
    # -------------------------
    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(root=os.path.join(dataset_dir, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Classes: {train_dataset.classes}")  # should print ['graph', 'random']

    # -------------------------
    # Model
    # -------------------------
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: graph vs random
    model = model.to(device)

    # -------------------------
    # Loss & Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------
    # Training Loop
    # -------------------------
    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
        val_acc = val_correct.double() / val_total

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet18_graph_classifier.pth")

    print("\n✅ Training finished!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("Model saved as 'best_resnet18_graph_classifier.pth'")

    # -------------------------
    # Optional: Test some predictions
    # -------------------------
    model.eval()
    import random
    from PIL import Image
    import matplotlib.pyplot as plt

    # pick 5 random images from val
    val_images = []
    for cls in ["graph","random"]:
        cls_folder = os.path.join(dataset_dir,"val",cls)
        val_images += [os.path.join(cls_folder,f) for f in os.listdir(cls_folder)]
    sample_imgs = random.sample(val_images,5)

    for img_path in sample_imgs:
        img = Image.open(img_path).convert("RGB")
        img_tensor = val_transforms(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        _, pred = torch.max(output,1)
        pred_class = train_dataset.classes[pred.item()]
        plt.imshow(img)
        plt.title(f"Predicted: {pred_class}")
        plt.axis("off")
        plt.show()


# -------------------------
# Windows multiprocessing safe entry
# -------------------------
if __name__ == "__main__":
    main()
