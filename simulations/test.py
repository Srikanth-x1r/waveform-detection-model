import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_MODEL_PATH = r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\functions_classifier.pth"

classes = [
    "linear", "quadratic", "exponential", "sinusoidal", "square",
    "triangular", "sawtooth", "sinc", "step_ramp", "gaussian", "piecewise"
]

# ---------------------------
# RANDOM SIGNAL GENERATOR
# ---------------------------
def generate_signal(cls, x):
    if cls == "linear":
        m = np.random.uniform(-5, 5)
        c = np.random.uniform(-3, 3)
        return m*x + c
    elif cls == "quadratic":
        a = np.random.uniform(-2, 2)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-3, 3)
        return a*x**2 + b*x + c
    elif cls == "exponential":
        a = np.random.uniform(-2, 2)
        return np.exp(a*x/5)
    elif cls == "sinusoidal":
        A = np.random.uniform(0.5, 2)
        f = np.random.uniform(0.5, 3)
        p = np.random.uniform(0, np.pi)
        return A * np.sin(2*np.pi*f*x + p)
    elif cls == "square":
        return np.sign(np.sin(2*np.pi*2*x))
    elif cls == "triangular":
        return 2*np.abs(2*(x-np.floor(x+0.5)))-1
    elif cls == "sawtooth":
        return 2*(x/np.max(x)-np.floor(0.5 + x/np.max(x)))
    elif cls == "sinc":
        return np.sinc(x)
    elif cls == "step_ramp":
        return np.where(x<0, 0, x)
    elif cls == "gaussian":
        mu = np.random.uniform(-1,1)
        sigma = np.random.uniform(0.2,1.0)
        return np.exp(-(x-mu)**2 / (2*sigma**2))
    elif cls == "piecewise":
        y = np.piecewise(x, [x<-1, (x>=-1) & (x<1), x>=1],
                         [lambda x: -1*x, lambda x: x**2, lambda x: np.sin(x)])
        return y

def save_signal_plot(cls):
    x = np.linspace(-5,5,400)
    y = generate_signal(cls, x)
    plt.figure(figsize=(3,3))
    plt.plot(x,y, color="blue")
    plt.grid(True, linewidth=0.3)
    plt.axis("on")
    img_path = "temp.png"
    plt.savefig(img_path, dpi=100, bbox_inches="tight")
    plt.close()
    return img_path, y

# ---------------------------
# LOAD MODEL
# ---------------------------
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------------------
# IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------------------
# TEST RANDOM GRAPH
# ---------------------------
true_class = np.random.choice(classes)
print(f"🟢 True function class: {true_class}")

img_path, true_values = save_signal_plot(true_class)
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = nn.Softmax(dim=1)(outputs)
    predicted_idx = probabilities.argmax().item()
    predicted_class = classes[predicted_idx]
    conf = probabilities[0][predicted_idx].item()

print(f"🤖 Predicted function class: {predicted_class} (Confidence: {conf*100:.2f}%)")

# ---------------------------
# SHOW DECISION DETAILS
# ---------------------------
print("\n--- Decision Tree / Stepwise Logic ---")
print(f"1. Input waveform image loaded ({img_path})")
print(f"2. Preprocessed to 224x224, normalized tensor")
print(f"3. Passed through pretrained ResNet18 backbone")
print(f"4. Features extracted by convolutional layers")
print(f"5. Flattened and passed through final fully-connected layer")
print(f"6. Softmax applied to get class probabilities")
for idx, cls in enumerate(classes):
    print(f"   Class '{cls}': {probabilities[0][idx].item()*100:.2f}%")
print(f"7. Predicted class = '{predicted_class}' (max probability)")

# ---------------------------
# SHOW GRAPH
# ---------------------------
plt.figure(figsize=(5,3))
plt.plot(np.linspace(-5,5,400), true_values, label=f"True: {true_class}")
plt.title(f"Predicted: {predicted_class}")
plt.grid(True)
plt.legend()
plt.show()
