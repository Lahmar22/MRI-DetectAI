# import numpy as np
# import tensorflow as tf
# from PIL import Image
from ultralytics import YOLO
from pathlib import Path
# IMAGE_SIZE = (512, 512)

# def predictGoogleNet(model_path, image_path, class_names):
#     # 1. Charger le modèle entraîné
#     model = tf.keras.models.load_model(model_path)

#     # 2. Charger et prétraiter l'image
#     image = tf.keras.utils.load_img(
#         image_path,
#         target_size=IMAGE_SIZE
#     )

#     image_array = tf.keras.utils.img_to_array(image)
#     image_array = image_array / 255.0

#     # Ajouter la dimension batch : (1, 512, 512, 3)
#     image_array = np.expand_dims(image_array, axis=0)

#     # 3. Effectuer la prédiction
#     predictions = model.predict(image_array, verbose=0)

#     # 4. Récupérer la classe prédite
#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = float(predictions[0][predicted_index]) * 100

#     print(f"Classe prédite : {predicted_class}")
#     print(f"Confiance : {confidence:.2f}%")

#     return {
#         "class": predicted_class,
#         "confidence": confidence,
#         "probabilities": predictions[0].tolist()
#     }

# class_names = [
#     "Glioma",
#     "Meningioma",
#     "No Tumor",
#     "Pituitary"
# ]

# result = predictGoogleNet(
#     model_path="models/googlenet_best.keras",
#     image_path="test.jpg",
#     class_names=class_names
# )

# print(result)

MODEL_PATH = "runs/detect/train/weights/best.pt"


def predict():
    # Vérifier que les fichiers existent
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    # if not Path(IMAGE_PATH).exists():
    #     raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH}")

    # Charger le modèle entraîné
    model = YOLO(MODEL_PATH)

    # Prédiction
    results = model.predict(
        source="test/",
        conf=0.25,
        imgsz=640,
        save=True
    )

    # Afficher les détections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(f"Classe : {model.names[class_id]}")
            print(f"Confiance : {confidence:.2%}")
            print(
                f"Boîte : x1={x1:.1f}, y1={y1:.1f}, "
                f"x2={x2:.1f}, y2={y2:.1f}"
            )

    print("Prédiction terminée.")

if __name__ == "__main__":
    predict()