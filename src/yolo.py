from ultralytics import YOLO
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent


def train_yolo():
    model = YOLO(BASE_DIR / "yolov8n.pt")
    data_yaml = BASE_DIR / "yolo" / "data.yaml"

    if torch.cuda.is_available():
        device = 0
        print("GPU disponible :", torch.cuda.get_device_name(0))
        print("Entraînement avec GPU CUDA")
    else:
        device = "cpu"
        print("GPU non disponible. Entraînement avec CPU")

    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=512,
        batch=8,
        patience=10,
        seed=42,
        save=True,
        save_period=10,
        device=device,
    )

    print("Entraînement terminé !")


if __name__ == "__main__":
    train_yolo()
