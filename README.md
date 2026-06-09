# West African Food Classifier API

A machine learning API for classifying West African food dishes from images using a trained ResNet18 model.

## 🍽️ Supported Food Classes

The model can classify **80+ West African food dishes** including:
- Jollof Rice, Fried Rice, Coconut Rice
- Egusi Soup, Okra Soup, Ogbono Soup, Banga Soup
- Fufu, Amala, Eba, Pounded Yam
- Akara, Chin Chin, Puff Puff, Bofrot
- Suya, Grilled Tilapia, Fish Roll
- And many more...

## ☁️ Cloud Storage with Cloudinary

**NEW**: You can now store your images in Cloudinary instead of bundling them with your deployment! This makes your Docker images much smaller and enables better scalability.

The API also supports deterministic class ordering via `class_names.json` (generated during training). If the cache file is missing, it will fall back to discovering class names from Cloudinary/local folders.

**Quick Setup:**
1. Upload images: `python upload_to_cloudinary.py` (see [CLOUDINARY_GUIDE.md](CLOUDINARY_GUIDE.md))
2. Set environment variables and run: `USE_CLOUDINARY=true python app.py`

📖 **Full guide**: See [CLOUDINARY_GUIDE.md](CLOUDINARY_GUIDE.md) for detailed instructions.

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Prepare environment variables:**
```bash
cp env.example .env
# edit .env with your Cloudinary creds or set USE_CLOUDINARY=false for local assets
```

2. **Build and run with Docker Compose (Cloudinary only):**
```bash
docker-compose up --build
```

> **Cloudinary only:** The compose file no longer mounts `images/` because inference pulls class names from Cloudinary. Ensure `USE_CLOUDINARY=true` and credentials are set in `.env`.

3. **Or build and run manually:**
```bash
docker build -t food-classifier .
docker run -p 8000:8000 --env-file .env food-classifier
```

### Option 2: Local Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the API:**
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Documentation
Visit `http://localhost:8000/docs` for interactive Swagger documentation.

### Endpoints

#### 1. Health Check
```http
GET /
```
Returns API status and basic information.

#### 2. Get All Classes
```http
GET /classes
```
Returns all available food classes.

#### 3. Predict from Image File
```http
POST /predict
Content-Type: multipart/form-data

file: [image file]
top_k: 5 (optional, default: 5)
```

#### 4. Predict from Multiple Images (Batch)
```http
POST /predict-batch
Content-Type: multipart/form-data

files: [image file 1, image file 2, ...]
top_k: 5 (optional, default: 5)
```

#### 5. Predict from Image URL
```http
POST /predict-url
Content-Type: application/json

{
  "image_url": "https://example.com/food.jpg",
  "top_k": 5
}
```

#### 6. Health Check
```http
GET /health
```
Detailed health check with model status.

## 🔧 Usage Examples

### Python Example

```python
import requests
import json

# Predict from file
with open('food_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['top_prediction']['class_name']}")
    print(f"Confidence: {result['top_prediction']['confidence_percentage']}%")

# Predict from URL
data = {
    "image_url": "https://example.com/jollof_rice.jpg",
    "top_k": 3
}
response = requests.post('http://localhost:8000/predict-url', json=data)
result = response.json()
```

### cURL Example

```bash
# Predict from file
curl -X POST "http://localhost:8000/predict" \
  -F "file=@food_image.jpg" \
  -F "top_k=5"

# Predict from URL
curl -X POST "http://localhost:8000/predict-url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/food.jpg", "top_k": 3}'
```

### JavaScript Example

```javascript
// Predict from file
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('top_k', '5');

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.top_prediction);
});
```

## 🐳 Docker Deployment

### Production Deployment

1. **Build production image:**
```bash
docker build -t food-classifier:latest .
```

2. **Run with production settings:**
```bash
docker run -d \
  --name food-classifier \
  -p 8000:8000 \
  -v $(pwd)/food_classifier.pth:/app/food_classifier.pth:ro \
  -v $(pwd)/images:/app/images:ro \
  --restart unless-stopped \
  food-classifier:latest
```

### Docker Compose for Production

```yaml
version: '3.8'
services:
  food-classifier:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./food_classifier.pth:/app/food_classifier.pth:ro
      - ./images:/app/images:ro
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## ☁️ Cloud Deployment Options

### 1. AWS EC2
- Launch EC2 instance with GPU support
- Install Docker
- Deploy using docker-compose
- Configure security groups for port 8000

### 2. Google Cloud Run
- Build and push to Google Container Registry
- Deploy to Cloud Run with GPU support
- Configure environment variables

### 3. Azure Container Instances
- Build and push to Azure Container Registry
- Deploy to Azure Container Instances
- Configure networking and storage

### 4. Heroku
- Add Dockerfile for Heroku deployment
- Configure Procfile
- Deploy using Heroku CLI

## 🔧 Configuration

### Environment Variables
- `MODEL_PATH`: Path to model file (default: "food_classifier.pth")
- `IMAGE_SIZE`: Input image size (default: 512)
- `DEVICE`: Device to use (auto-detected)

### Performance Optimization
- Use GPU if available (automatically detected)
- Adjust batch size for your hardware
- Consider model quantization for faster inference

## 📊 Model Information

- **Architecture**: ResNet18
- **Input Size**: 512x512 pixels
- **Classes**: 80+ West African food dishes
- **Framework**: PyTorch
- **Preprocessing**: ImageNet normalization

## 🛠️ Development

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test with sample images
python test.py test/j1.jpeg
```

## 📝 API Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "class_name": "jollof_rice",
      "confidence": 0.95,
      "confidence_percentage": 95.0
    },
    {
      "class_name": "fried_rice", 
      "confidence": 0.03,
      "confidence_percentage": 3.0
    }
  ],
  "top_prediction": {
    "class_name": "jollof_rice",
    "confidence": 0.95,
    "confidence_percentage": 95.0
  }
}
```

## 🚨 Error Handling

The API includes comprehensive error handling for:
- Invalid image formats
- Model loading errors
- Network timeouts
- File size limits
- Malformed requests

## 📈 Monitoring

- Health check endpoint: `/health`
- Logging configured for production
- Docker health checks included
- Metrics available via FastAPI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the health check at `/health`
3. Check Docker logs: `docker logs food-classifier`
4. Verify model file exists and is accessible
