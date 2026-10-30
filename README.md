# MRI DetectAI 

## Run with Docker Compose

From the project root, start both services:

```powershell
docker compose up --build -d
```

- Streamlit: http://localhost:8501
- Airflow: http://localhost:8080

## MRI predictions

In the Streamlit dashboard, choose GoogLeNet or ResNet50 for a four-class
classification result, or YOLO to view tumor detections and their confidence
scores. Upload a JPG, PNG, BMP, or WebP MRI image and select **Run prediction**.
The dashboard uses the trained models in `models/` and
`runs/detect/train/weights/best.pt`.

Predictions are for research and educational use only; they are not a medical
diagnosis.

To inspect service logs or stop the containers:

```powershell
docker compose logs -f airflow
docker compose logs -f streamlit
docker compose down
```