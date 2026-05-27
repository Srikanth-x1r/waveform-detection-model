import torch
from torchvision import models, transforms
import cv2
from PIL import Image

# -------------------------
# Load model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(r"S:\Srikanth\VIT\Academics\pythonProject\Signals_project\best_resnet18_graph_classifier.pth", map_location=device))
model.to(device)
model.eval()

classes = ["graph", "random"]

# -------------------------
# Transform (same as training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Open camera
# -------------------------
cap = cv2.VideoCapture(0)  # 0 is default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR frame to PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    # Transform and add batch dimension
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = classes[pred.item()]

    # Overlay prediction on frame
    cv2.putText(frame, f"Prediction: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Graph Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
