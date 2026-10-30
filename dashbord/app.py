from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = (512, 512)
CLASSIFICATION_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASSIFICATION_MODELS = {
    "GoogLeNet": PROJECT_ROOT / "models" / "googlenet_best.keras",
    "ResNet50": PROJECT_ROOT / "models" / "resnet50_best.keras",
}
YOLO_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"


st.set_page_config(
    page_title="MRI DetectAI",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_classification_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
        },
    )


@st.cache_resource
def load_yolo_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def prepare_classification_image(image: Image.Image) -> np.ndarray:
    import tensorflow as tf

    image_array = tf.convert_to_tensor(np.asarray(image, dtype=np.float32))
    resized_image = tf.image.resize(image_array, IMAGE_SIZE)
    return resized_image.numpy()[np.newaxis, ...]


def display_classification_result(model_name: str, image: Image.Image) -> None:
    model_path = CLASSIFICATION_MODELS[model_name]
    model = load_classification_model(str(model_path))
    image_batch = prepare_classification_image(image)
    predictions = np.asarray(model.predict(image_batch, verbose=0))

    expected_shape = (1, len(CLASSIFICATION_CLASSES))
    if predictions.shape != expected_shape:
        raise ValueError(
            f"{model_name} returned predictions with shape {predictions.shape}; "
            f"{expected_shape} was expected."
        )

    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASSIFICATION_CLASSES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader("Classification result")
    metric, confidence_metric = st.columns(2)
    metric.metric("Predicted class", predicted_class)
    confidence_metric.metric("Confidence", f"{confidence:.1%}")
    st.caption(
        "Class order: Glioma, Meningioma, No Tumor, Pituitary. "
        "Scores are model confidence estimates, not a medical diagnosis."
    )
    st.bar_chart(
        {
            "Probability": {
                name: float(probabilities[index])
                for index, name in enumerate(CLASSIFICATION_CLASSES)
            }
        },
        y_label="Probability",
    )


def display_yolo_result(image: Image.Image) -> None:
    model = load_yolo_model(str(YOLO_MODEL_PATH))
    results = model.predict(source=image, imgsz=512, conf=0.25, verbose=False)
    result = results[0]
    boxes = result.boxes

    st.subheader("YOLO detection result")
    annotated_bgr = result.plot()
    st.image(annotated_bgr[:, :, ::-1], caption="Detections", use_container_width=True)

    if boxes is None or len(boxes) == 0:
        st.info("No tumor detections were found above the 25% confidence threshold.")
        return

    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    names = model.names
    detections = [
        {
            "Class": str(names[class_id]).replace("_", " "),
            "Confidence": float(confidence),
        }
        for class_id, confidence in zip(class_ids, confidences)
    ]
    detections.sort(key=lambda detection: detection["Confidence"], reverse=True)

    st.metric("Detections", len(detections))
    st.dataframe(
        [
            {
                "Class": detection["Class"],
                "Confidence": f"{detection['Confidence']:.1%}",
            }
            for detection in detections
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("YOLO reports localized detections; no detection does not rule out disease.")


st.title("MRI DetectAI")
st.write(
    "Upload an MRI image to classify it with GoogLeNet or ResNet50, "
    "or localize tumors with YOLO."
)
st.warning(
    "For research and educational use only. This tool is not a medical device "
    "and must not be used to diagnose or guide treatment."
)

with st.sidebar:
    st.header("Prediction settings")
    selected_model = st.radio(
        "Model",
        ("GoogLeNet", "ResNet50", "YOLO"),
        help="Choose a trained image classifier or object-detection model.",
    )
    st.caption("Classification input size: 512 × 512 pixels.")
    if selected_model == "YOLO":
        st.caption("YOLO confidence threshold: 25%.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=("jpg", "jpeg", "png", "bmp", "webp"),
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Could not read the uploaded image: {error}")
    else:
        image_column, result_column = st.columns([1, 1])
        with image_column:
            st.image(uploaded_image, caption="Uploaded MRI", use_container_width=True)

        model_path = (
            YOLO_MODEL_PATH
            if selected_model == "YOLO"
            else CLASSIFICATION_MODELS[selected_model]
        )
        if not model_path.is_file():
            st.error(f"Model file not found: {model_path}")
        elif st.button("Run prediction", type="primary", use_container_width=True):
            with result_column:
                with st.spinner(f"Running {selected_model} prediction..."):
                    if selected_model == "YOLO":
                        display_yolo_result(uploaded_image)
                    else:
                        display_classification_result(selected_model, uploaded_image)
