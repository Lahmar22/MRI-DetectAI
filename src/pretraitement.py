import tensorflow as tf
from tensorflow.keras import layers, Model, models

from separation import separeDataTrain
from app import getImages
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32)


def preprocess_image(image_path, label):
    image = load_image(image_path)
    return image, label


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


def visualiser_image_augmented():
    import matplotlib.pyplot as plt

    images = next(
        (item["images"] for item in getImages() if item["images"]),
        None,
    )
    if images is None:
        raise ValueError("Aucune image disponible pour la visualisation.")

    image = load_image(str(images[0]))
    augmented_image = data_augmentation(
        tf.expand_dims(image, axis=0),
        training=True,
    )[0]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.numpy())
    plt.title("Image originale")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image.numpy())
    plt.title("Image augmentée")
    plt.axis("off")

    plt.show()


def pipelineData():
    x_train, x_test, y_train, y_test, x_val, y_val = separeDataTrain()

    class_names = sorted(set(y_train))

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    y_train = [class_to_index[name] for name in y_train]
    y_test = [class_to_index[name] for name in y_test]
    y_val = [class_to_index[name] for name in y_val]

    print("Classes :", class_names)
    print("X_train :", len(x_train))
    print("Y_train :", y_train[:10])
    print("X_test :", len(x_test))
    print("Y_test :", y_test[:10])
    print("X_val :", len(x_val))
    print("Y_val :", y_val[:10])

    def create_dataset(images, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (list(map(str, images)), labels)
        )
        ds = ds.map(
            preprocess_image,
            num_parallel_calls=AUTOTUNE
        )
        if training:
            ds = ds.shuffle(1000)
        ds = ds.batch(BATCH_SIZE)
        if training:
            ds = ds.map(
                lambda images, labels: (
                    data_augmentation(images, training=True),
                    labels,
                ),
                num_parallel_calls=AUTOTUNE,
            )
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = create_dataset(x_train, y_train, training=True)
    test_ds = create_dataset(x_test, y_test)
    val_ds = create_dataset(x_val, y_val)

    return train_ds, test_ds, val_ds, class_names


def googleeNet():
    train_ds, _, val_ds, class_names = pipelineData()
    num_classes = len(class_names)

    base_model = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.applications.inception_v3.preprocess_input(
        inputs
    )

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    base_model.trainable = True

    # Garder gelées la majorité des couches
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Garder les BatchNormalization gelées
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        filepath="models/googlenet_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        save_weights_only=False,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        mode="min",
        min_lr=1e-7,
        verbose=1
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

def resNet(num_classes, image_size=(512, 512)):
    
    inputs = layers.Input(
        shape=(*image_size, 3),
        name="mri_image"
    )

    # ResNet50 preprocessing
    x = layers.Lambda(
        preprocess_input,
        name="resnet_preprocessing"
    )(inputs * 255.0)

    # Pretrained backbone
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*image_size, 3)
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = models.Model(inputs, outputs, name="MRI_ResNet50")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def resNetTrin():
    train_ds, _, val_ds, class_names = pipelineData()
    num_classes = len(class_names)

    model = resNet(num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/resnet50_best.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-3
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )

    model.save("models/resnet50_best.keras")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("GPU")
        # googleeNet()
        resNetTrin()
    else:
        print("⚠️ Aucun GPU détecté")
        resNetTrin()