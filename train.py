import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------ CONFIG ------------------
IMG_SIZE = 256
BATCH = 32
EPOCHS = 50
PATIENCE = 6
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ DATA ------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/", transform=transform)
train_sz = int(0.8 * len(dataset))
val_sz = len(dataset) - train_sz

train_ds, val_ds = random_split(
    dataset, [train_sz, val_sz],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader   = DataLoader(val_ds, BATCH, shuffle=False)

# ------------------ MODEL ------------------
model = nn.Sequential(
    nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64,128,3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128*30*30, 1),
    nn.Sigmoid()
).to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

# ------------------ TRAIN ------------------
best_acc, wait = 0, 0
train_accs, val_accs, losses = [], [], []

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    correct = total = running_loss = 0

    for x,y in train_loader:
        x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        correct += ((out > 0.5) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    losses.append(running_loss / len(train_loader))
    train_accs.append(train_acc)

    # ---- VALIDATE ----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            correct += ((out > 0.5).squeeze() == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:02d} | Train {train_acc:.3f} | Val {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping")
            break

# ------------------ PLOTS ------------------
import matplotlib.pyplot as plt
plt.plot(train_accs,label="Train")
plt.plot(val_accs,label="Val")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.show()

plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

# ------------------ CONF MATRIX ------------------
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x,y in val_loader:
        out = model(x.to(DEVICE))
        y_true += y.tolist()
        y_pred += (out>0.5).int().cpu().squeeze().tolist()

ConfusionMatrixDisplay(confusion_matrix(y_true,y_pred)).plot()
plt.show()



