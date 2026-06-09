import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os
import torch.nn.functional as F

MODEL_PATH = "food_classifier.pth"
CLASS_NAMES = sorted(os.listdir("images"))
IMAGE_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, top_k=1):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(top_k, dim=1)

        print("🍽️ Top Predictions:")
        for prob, idx in zip(top_probs[0], top_idxs[0]):
            class_name = CLASS_NAMES[idx.item()]
            confidence = prob.item() * 100
            print(f"✅ {class_name}: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image.jpg")
    else:
        predict(sys.argv[1], top_k=3)

