import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "slope_intercept_model2.pth"  # your trained model
IMG_SIZE = (224, 224)

# --------------------
# GENERATE RANDOM LINEAR GRAPH
# --------------------
m_actual = np.random.uniform(-5, 5)   # random slope
c_actual = np.random.uniform(-10, 10) # random intercept
x = np.linspace(-10, 10, 100)
y = m_actual * x + c_actual

plt.figure(figsize=(4, 4))
plt.plot(x, y, color="blue", linewidth=2)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.grid(True)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Save the figure temporarily
plt.savefig("temp_linear.png")
plt.close()

# --------------------
# LOAD IMAGE AND TRANSFORM
# --------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

img = Image.open("temp_linear.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# --------------------
# LOAD MODEL
# --------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # output: slope and intercept
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --------------------
# PREDICT
# --------------------
with torch.no_grad():
    pred = model(img_tensor).cpu().numpy()[0]

m_pred, c_pred = pred
print(f"Actual slope (m): {m_actual:.4f}, Predicted slope (m): {m_pred:.4f}")
print(f"Actual intercept (c): {c_actual:.4f}, Predicted intercept (c): {c_pred:.4f}")

# --------------------
# SHOW GRAPH
# --------------------
plt.figure(figsize=(4,4))
plt.plot(x, y, color="blue", linewidth=2, label="Generated Graph")
plt.title(f"Predicted m={m_pred:.2f}, c={c_pred:.2f}")
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.grid(True)
plt.legend()
plt.show()
