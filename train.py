from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm
import json

DATA_DIR = "images"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_acc = correct / total * 100
    print(f"Loss: {running_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

torch.save(model.state_dict(), "food_classifier.pth")
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(dataset.classes, f, ensure_ascii=False, indent=2)
print("✅ Training complete and model saved.")

