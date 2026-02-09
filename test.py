import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


model = nn.Sequential(
    nn.Conv2d(3,32,3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64,128,3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128*30*30,1),
    nn.Sigmoid()
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Load test data
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

test_ds = datasets.ImageFolder("new_test_data/", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 2️⃣ Load trained model
model.load_state_dict(torch.load("best_model.pt"))
model.to(DEVICE)
model.eval()

# 3️⃣ Evaluate
correct = total = 0
y_true, y_pred = [], []

with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        preds = (out > 0.5).int().squeeze()

        correct += (preds == y).sum().item()
        total += y.size(0)

        y_true += y.tolist()
        y_pred += preds.cpu().tolist()

print("✅ Test Accuracy:", correct / total)

# 4️⃣ Confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
plt.show()
