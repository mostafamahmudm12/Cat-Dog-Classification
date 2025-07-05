# Image Classification API & Web App

A complete image classification solution built with FastAPI and Streamlit, featuring a CNN model for binary classification (Dogs vs Cats). The project provides both a RESTful API and a user-friendly web interface for batch image classification.

## 🏗️ Project Structure

```
├── .env                          # Environment variables
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore file
├── main.py                       # FastAPI application
├── streamapp.py                  # Streamlit web interface
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/
│   ├── config.py                 # Configuration settings
│   ├── inference.py              # Image classifier implementation
│   ├── schemas.py                # Pydantic models
│   │
│   ├── assets/
│   │   ├── idx2label.joblib      # Label mappings
│   │   ├── model.keras           # Trained CNN model
│   │   │
│   │   └── downloaded-images/    # Temporary storage for uploaded images
│   │       └── dog.10073.jpg     # Sample image
│   │
│   ├── notebooks/
│   │   └── 03_CNN_dogs_vs__cats.ipynb  # Model training notebook
│   │
│   └── __pycache__/              # Python cache files
│
└── __pycache__/                  # Python cache files
```

## 🚀 Features

### FastAPI Backend
- **RESTful API** with automatic OpenAPI documentation
- **Batch Processing** support for multiple images
- **Two Classification Methods**:
  - Memory-based (for small batches)
  - File-based (for large batches with automatic cleanup)
- **API Key Authentication** for secure access
- **CORS Support** for cross-origin requests
- **Comprehensive Error Handling** and logging

### Streamlit Web Interface
- **User-friendly Upload Interface** with drag-and-drop support
- **Real-time Image Preview** for uploaded files
- **Interactive Results Dashboard** with confidence scores
- **Data Visualizations**:
  - Confidence distribution histograms
  - Predicted classes pie charts
  - Processing time metrics
- **Export Functionality** (JSON download)
- **API Configuration Panel** with connection testing

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd image-classification-api
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env file with your configuration
```

4. **Create required directories**:
```bash
mkdir -p src/assets/downloaded-images
```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Application Settings
APP_NAME=Image Classification API
VERSION=1.0.0

# Security
API_SECRET_KEY=your-secret-api-key-here

# Storage
DOWNLOADED_IMAGES_FOLDER=src/assets/downloaded-images

# Model Settings
MODEL_PATH=src/assets/model.keras
LABELS_PATH=src/assets/idx2label.joblib
```

### Required Files
- `src/assets/model.keras`: Trained CNN model
- `src/assets/idx2label.joblib`: Label mappings file
- Ensure the downloaded-images folder exists for temporary storage

## 🔧 Usage

### Running the FastAPI Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Running the Streamlit Web App

```bash
streamlit run streamapp.py
```

The web interface will be available at `http://localhost:8501`

## 📡 API Endpoints

### Authentication
All endpoints require an API key passed in the `X-API-Key` header.

### Available Endpoints

#### 1. Health Check
```http
GET /
```
**Headers**: `X-API-Key: your-api-key`

**Response**:
```json
{
  "app_name": "Image Classification API",
  "version": "1.0.0",
  "status": "up & running"
}
```

#### 2. Batch Classification (Memory-based)
```http
POST /classify-batch-memory
```
**Headers**: `X-API-Key: your-api-key`

**Body**: `multipart/form-data` with image files

**Use Case**: Small batches (< 10 images), faster processing

#### 3. Batch Classification (File-based)
```http
POST /classify-batch-paths
```
**Headers**: `X-API-Key: your-api-key`

**Body**: `multipart/form-data` with image files

**Use Case**: Large batches, automatic file cleanup

### Response Format
```json
{
  "predictions": [
    {
      "filename": "image1.jpg",
      "predicted_class": "dog",
      "confidence": 0.95,
      "probabilities": {
        "dog": 0.95,
        "cat": 0.05
      }
    }
  ],
  "processing_time": 1.23,
  "total_images": 1
}
```

## 🐳 Docker Support

### Build Docker Image
```bash
docker build -t image-classification-api .
```

### Run with Docker
```bash
docker run -p 8000:8000 -e API_SECRET_KEY=your-secret-key image-classification-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_SECRET_KEY=your-secret-key
    volumes:
      - ./src/assets:/app/src/assets
```

## 🧪 Testing

### Using curl
```bash
# Test API connection
curl -X GET "http://localhost:8000/" -H "X-API-Key: your-api-key"

# Test image classification
curl -X POST "http://localhost:8000/classify-batch-memory" \
  -H "X-API-Key: your-api-key" \
  -F "files=@path/to/image1.jpg" \
  -F "files=@path/to/image2.jpg"
```

### Using Python requests
```python
import requests

# API configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY}

# Test connection
response = requests.get(f"{API_BASE_URL}/", headers=headers)
print(response.json())

# Classify images
files = [
    ("files", ("image1.jpg", open("image1.jpg", "rb"), "image/jpeg")),
    ("files", ("image2.jpg", open("image2.jpg", "rb"), "image/jpeg"))
]
response = requests.post(
    f"{API_BASE_URL}/classify-batch-memory",
    files=files,
    headers=headers
)
print(response.json())
```

## 📊 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Task**: Binary Classification (Dogs vs Cats)
- **Input**: RGB images (automatically resized)
- **Output**: Class probabilities and predictions
- **Training**: See `src/notebooks/03_CNN_dogs_vs__cats.ipynb`

## 🔒 Security Features

- **API Key Authentication**: Secure endpoint access
- **Input Validation**: File type and content verification
- **Error Handling**: Comprehensive error responses
- **CORS Configuration**: Configurable cross-origin access
- **File Cleanup**: Automatic temporary file deletion

## 📈 Performance Considerations

### Memory-based Classification
- ✅ **Faster processing** for small batches
- ✅ **No disk I/O** overhead
- ❌ **Higher memory usage**
- 🎯 **Recommended**: < 10 images or total size < 50MB

### File-based Classification
- ✅ **Lower memory usage**
- ✅ **Better for large batches**
- ✅ **Automatic cleanup**
- ❌ **Disk I/O overhead**
- 🎯 **Recommended**: > 10 images or total size > 50MB

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error (403)**:
   - Check your API key in the `.env` file
   - Ensure the `X-API-Key` header is included

2. **Model Not Found**:
   - Verify `src/assets/model.keras` exists
   - Check the `MODEL_PATH` in configuration

3. **Upload Directory Error**:
   - Ensure `src/assets/downloaded-images/` directory exists
   - Check write permissions

4. **Memory Issues**:
   - Use file-based classification for large batches
   - Reduce batch size for memory-based processing

### Logging
The application uses Python's logging module. Check console output for detailed error messages and processing information.

## 📝 Development

### Adding New Models
1. Train your model and save as `.keras` format
2. Update label mappings in `idx2label.joblib`
3. Modify `src/inference.py` if needed
4. Update model path in configuration

### Extending the API
1. Add new endpoints in `main.py`
2. Update schemas in `src/schemas.py`
3. Modify the Streamlit interface as needed
4. Update documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **TensorFlow/Keras**: https://tensorflow.org/

## 📧 Support

For support, please open an issue in the repository or contact the development team.

---

**Built with ❤️ using FastAPI and Streamlit**