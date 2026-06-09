from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import io
import os
import uvicorn
from typing import List, Dict
import logging
from dotenv import load_dotenv
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="West African Food Classifier API",
    description="API for classifying West African food dishes from images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "food_classifier.pth"
IMAGE_SIZE = 512
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")

USE_CLOUDINARY = os.getenv("USE_CLOUDINARY", "false").lower() == "true"
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "food-classifier")  # Folder prefix in Cloudinary

def get_class_names():
    """Get class names from Cloudinary or local directory"""
    if USE_CLOUDINARY and CLOUDINARY_CLOUD_NAME:
        try:
            import cloudinary
            import cloudinary.api
            import cloudinary.exceptions

            cloudinary.config(
                cloud_name=CLOUDINARY_CLOUD_NAME,
                api_key=CLOUDINARY_API_KEY,
                api_secret=CLOUDINARY_API_SECRET
            )

            classes = set()
            folder_prefix = CLOUDINARY_FOLDER.strip("/") if CLOUDINARY_FOLDER else ""

            next_cursor = None
            try:
                while True:
                    if folder_prefix:
                        response = cloudinary.api.subfolders(folder_prefix, next_cursor=next_cursor)
                    else:
                        response = cloudinary.api.root_folders(next_cursor=next_cursor)

                    for folder in response.get("folders", []):
                        name = folder.get("name")
                        if name:
                            classes.add(name.strip().split("/")[-1])

                    next_cursor = response.get("next_cursor")
                    if not next_cursor:
                        break

            except (cloudinary.exceptions.NotFound, cloudinary.api.NotFound):
                # Folder might not exist yet; fall back to resource listing
                classes.clear()
            except Exception as subfolder_error:
                logger.debug(f"Cloudinary subfolder lookup failed: {subfolder_error}. Falling back to resource listing.")
                classes.clear()

            # Fall back to listing resources if no classes were found via subfolders
            if not classes:
                next_cursor = None
                prefix = f"{folder_prefix}/" if folder_prefix else ""

                while True:
                    response = cloudinary.api.resources(
                        type="upload",
                        prefix=prefix,
                        max_results=500,
                        next_cursor=next_cursor
                    )

                    for resource in response.get("resources", []):
                        parts = resource.get("public_id", "").split("/")
                        if folder_prefix:
                            if len(parts) >= 2:
                                classes.add(parts[1])
                        elif len(parts) >= 1:
                            classes.add(parts[0])

                    next_cursor = response.get("next_cursor")
                    if not next_cursor:
                        break

            if classes:
                sorted_classes = sorted(list(classes))
                logger.info(f"✅ Loaded {len(sorted_classes)} classes from Cloudinary")
                return sorted_classes
            else:
                logger.warning("No classes found in Cloudinary, falling back to local")
        except Exception as e:
            logger.warning(f"Cloudinary error: {e}, falling back to local storage")

    # Fallback to local directory
    if os.path.exists("images"):
        classes = sorted(os.listdir("images"))
        logger.info(f"✅ Loaded {len(classes)} classes from local directory")
        return classes

    return []

def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and all(isinstance(x, str) for x in cached) and cached:
                logger.info(f"✅ Loaded {len(cached)} classes from cache: {CLASS_NAMES_PATH}")
                return cached
            logger.warning(f"Cache file {CLASS_NAMES_PATH} is invalid. Falling back to discovery.")
        except Exception as e:
            logger.warning(f"Failed to load cache {CLASS_NAMES_PATH}: {e}. Falling back to discovery.")

    discovered = get_class_names()
    logger.info(f"✅ Loaded {len(discovered)} classes from discovery source")
    return discovered

CLASS_NAMES = load_class_names()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class FoodClassifier:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            if not CLASS_NAMES:
                raise ValueError("No class names found. Make sure 'images' directory exists.")
            
            self.model = models.resnet18(weights=None)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
            
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully with {len(CLASS_NAMES)} classes")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Make prediction on an image"""
        try:
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                top_probs, top_idxs = probs.topk(top_k, dim=1)
            
            results = []
            for prob, idx in zip(top_probs[0], top_idxs[0]):
                results.append({
                    "class_name": CLASS_NAMES[idx.item()],
                    "confidence": float(prob.item()),
                    "confidence_percentage": round(float(prob.item()) * 100, 2)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

classifier = FoodClassifier()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "West African Food Classifier API",
        "status": "healthy",
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES[:10] + ["..."] if len(CLASS_NAMES) > 10 else CLASS_NAMES
    }

@app.get("/classes")
async def get_classes():
    """Get all available food classes"""
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict_food(file: UploadFile = File(...), top_k: int = 5):
    """
    Predict the food class from an uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, webp)
        top_k: Number of top predictions to return (default: 5)
    
    Returns:
        JSON with predictions and confidence scores
    """
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        if not file.content_type:
            logger.debug(f"Upload missing content_type header. File name: {file.filename}")
        elif not file.content_type.startswith('image/'):
            logger.debug(f"Upload content_type '{file.content_type}' does not start with image/. Proceeding based on PIL validation.")

        predictions = classifier.predict(image, top_k=top_k)

        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Form(5),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: List[Dict] = []
    for file in files:
        try:
            image_data = await file.read()
            if not image_data:
                raise ValueError("Uploaded file is empty")

            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception:
                raise ValueError("Uploaded file is not a valid image")

            predictions = classifier.predict(image, top_k=top_k)
            results.append({
                "filename": file.filename,
                "success": True,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,
            })
        except Exception as e:
            logger.error(f"Batch prediction error for {getattr(file, 'filename', 'unknown')}: {str(e)}")
            results.append({
                "filename": getattr(file, "filename", None),
                "success": False,
                "error": str(e),
                "predictions": [],
                "top_prediction": None,
            })

    return {
        "success": True,
        "count": len(results),
        "results": results,
    }

@app.post("/predict-url")
async def predict_from_url(image_url: str, top_k: int = 5):
    """
    Predict food class from image URL
    
    Args:
        image_url: URL of the image
        top_k: Number of top predictions to return
    
    Returns:
        JSON with predictions and confidence scores
    """
    try:
        import requests
        
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        predictions = classifier.predict(image, top_k=top_k)
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        }
        
    except Exception as e:
        logger.error(f"URL prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "device": str(classifier.device),
        "total_classes": len(CLASS_NAMES),
        "model_path": MODEL_PATH,
        "storage": "cloudinary" if USE_CLOUDINARY and CLOUDINARY_CLOUD_NAME else "local",
        "cloudinary_configured": USE_CLOUDINARY and bool(CLOUDINARY_CLOUD_NAME)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
