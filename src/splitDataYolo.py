from app import getImages
from separation import getDataVal
from pathlib import Path
import shutil
import random

base_dir = Path(__file__).resolve().parent


dataVal = getDataVal()
dataTrain = getImages()


dataset = Path("yolo")


random.seed(42)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_directories():
    for split in ["train", "val", "test"]:
        (dataset / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (dataset / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )


def split_data_yolo(data, split_name, train_ratio=0.8):
    
    images_dest = dataset / "images" / split_name
    labels_dest = dataset / "labels" / split_name

    copied_images = 0
    copied_labels = 0
    missing_labels = 0

    for item in data:
        images = item["images"]
        labels = item["labels"]

        labels_by_stem = {
            Path(label).stem: Path(label)
            for label in labels
        }

        pairs = []

        for image in images:
            image = Path(image)

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = labels_by_stem.get(image.stem)

            if label is None or not label.exists():
                print(f"Label manquant : {image}")
                missing_labels += 1
                continue

            pairs.append((image, label))


        random.shuffle(pairs)


        if split_name == "train_test":
            split_index = int(len(pairs) * train_ratio)
            train_pairs = pairs[:split_index]
            test_pairs = pairs[split_index:]

            destinations = [
                ("train", train_pairs),
                ("test", test_pairs),
            ]
        else:
            destinations = [(split_name, pairs)]

        for target_split, split_pairs in destinations:
            img_dir = dataset / "images" / target_split
            lbl_dir = dataset / "labels" / target_split

            for image, label in split_pairs:
                # Préserver les noms des fichiers
                shutil.copy2(image, img_dir / image.name)
                shutil.copy2(label, lbl_dir / label.name)

                copied_images += 1
                copied_labels += 1

    print(f"\nSplit : {split_name}")
    print(f"Images copiées : {copied_images}")
    print(f"Labels copiés : {copied_labels}")
    print(f"Labels manquants : {missing_labels}")


prepare_directories()

split_data_yolo(dataTrain, "train_test", train_ratio=0.8)

split_data_yolo(dataVal, "val")

print("\nDataset YOLOv8 créé dans :", dataset.resolve())
