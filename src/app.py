from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import Counter

base_dir = Path(__file__).resolve().parent

dataSet = (base_dir / ".." / "data" / "Train").resolve()  

def getImages():
    data = []
    for classe in dataSet.iterdir():
        if classe.is_dir():
            images_dir = classe / "images"
            labels_dir = classe / "labels"
            Images = list(images_dir.glob("*"))
            labels = list(labels_dir.glob("*"))
            data.append({
                "classe": classe.name,
                "images": Images,
                "labels": labels
            })
    return data

data = getImages()

# for item in data:
#     print(item["classe"])
#     print(len(item["images"]))
#     print(len(item["labels"]))



def visualiser(data):
    classes = [item["classe"] for item in data]
    total_images = [len(item["images"]) for item in data]
    total_labels = [len(item["labels"]) for item in data]

    x = np.arange(len(classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, total_images, width, label="Images", color="steelblue")
    ax.bar(x + width / 2, total_labels, width, label="Labels", color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Images and labels per class")
    ax.legend()

    plt.tight_layout()
    plt.show()

# visualiser(data)

def showImages(data):

    plt.figure(figsize=(10, 6))

    for i, item in enumerate(data):

        image = Image.open(item["images"][0])
        plt.subplot(1, len(data), i+1)
        plt.imshow(image)
        plt.title(item["classe"])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# showImages(data)

def infoImage(data):
    for item in data:
        dimensions = Counter()
        formats = Counter()
        modes = Counter()
        erreurs = 0
        for chemin_image in item["images"]:
            try:
                with Image.open(chemin_image) as image:
                    dimensions[image.size] += 1
                    formats[image.format] += 1
                    modes[image.mode] += 1 
            except Exception:
                erreurs += 1
        print(f"\nClasse : {item['classe']}")
        print("Dimensions :", dimensions)
        print("Formats :", formats)
        print("Modes :", modes)
        print("Images illisibles :", erreurs)


# infoImage(data)

def validate_label(data, num_classes=None):
            errors = []
            for item in data:
                for chemionLabels in item["labels"]:
                    with open(chemionLabels, "r") as file:
                        for line_number, line in enumerate(file, start=1):
                            values = line.split()
            
                            if len(values) != 5:
                                errors.append(
                                    f"Ligne {line_number}: 5 valeurs attendues"
                                )
                                continue
            
                            try:
                                class_id = int(values[0])
                                x, y, w, h = map(float, values[1:])
                            except ValueError:
                                errors.append(
                                    f"Ligne {line_number}: valeurs invalides"
                                )
                                continue
            
                            if class_id < 0 or (
                                num_classes is not None and class_id >= num_classes
                            ):
                                errors.append(
                                    f"Ligne {line_number}: classe invalide"
                                )
            
                            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                                errors.append(
                                    f"Ligne {line_number}: coordonnées hors limites"
                                )
            
                            if w <= 0 or h <= 0:
                                errors.append(
                                    f"Ligne {line_number}: largeur ou hauteur invalide"
                                )
            return errors

# print(validate_label(data))

