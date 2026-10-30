from ultralytics import YOLO


def evaluate_yolo():
    model = YOLO("runs/detect/train/weights/best.pt")

    metrics = model.val(
        data="yolo/data.yaml",
        split="val",
        imgsz=512,
        batch=8,
        plots=True
    )

    print("mAP@50 :", metrics.box.map50)
    print("mAP@50-95 :", metrics.box.map)
    print("Precision :", metrics.box.mp)
    print("Recall :", metrics.box.mr)


if __name__ == "__main__":
    evaluate_yolo()