from sklearn.model_selection import train_test_split
from pathlib import Path
from app import getImages

base_dir = Path(__file__).resolve().parent

dataSetVal = (base_dir / ".." / "data" / "Val").resolve()



def getDataVal():
    data = []
    for classe in dataSetVal.iterdir():
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


def separeDataTrain():
    dataTrain = getImages()
    dataVal = getDataVal()
    X = []
    y = []
    x_val = []
    y_val = []
    for item in dataVal:
        for imagename in item["images"]:
            x_val.append(imagename)
            y_val.append(item["classe"])

    for item in dataTrain:
        for imagename in item["images"]:
            X.append(imagename)
            y.append(item["classe"])

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True, stratify=y)
    
    return x_train, x_test, y_train, y_test, x_val, y_val

     


















# pairsTrain = []
# for item in dataTrain:
#     label_map = {Path(p).stem: Path(p) for p in item["labels"]}
#     for img in item["images"]: 
#         img = Path(img)
#         if img.stem in label_map:
#             pairsTrain.append((img, label_map[img.stem]))