"""This script is used to train a model on the dataset in the data/ folder."""

import os
import random
import time
import PIL
from matplotlib import pyplot as plt
import psutil
import tensorflow as tf


from pynvml import nvmlInit

IMAGE_SHAPE = (224, 224, 3)
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DATA_DIR = "data/"
USE_CUDA = True


print("TensorFlow version: " + tf.__version__)
print("Pillow version: " + PIL.__version__)

process = psutil.Process(os.getpid())

if tf.test.is_built_with_cuda():
    print("CUDA is available")
    CUDA = True
    nvmlInit()
else:
    print("CUDA is NOT available")
    CUDA = False

# if use_cuda is set to False disable GPU
if USE_CUDA and CUDA:
    print("Using GPU")
else:
    print("Using CPU")
    CUDA = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def make_dataset():
    """Create a tf.data.Dataset from the data in the data/ folder."""
    start_time = time.time()

    rand_seed = random.randint(0, 1000)

    print("Loading images from " + DATA_DIR + " ...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        shuffle=True,
        seed=rand_seed,
        interpolation="bilinear",
    )

    class_names = dataset.class_names
    print("Class names: " + str(class_names))

    train_size = int(0.8 * len(dataset))

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    print("Training dataset size: " + str(len(train_ds)))
    print("Validation dataset size: " + str(len(val_ds)))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    print(
        "Loading complete after " + str(round(time.time() - start_time, 2)) + " seconds"
    )

    example_images = val_ds.take(9)
    # show 9 example images and labels
    # for images, labels in example_images:
    #     for i in range(9):
    #         image = images[i].numpy().astype("uint8")
    #         label = labels[i].numpy()
    #         label = label.argmax()
    #         label = class_names[label]
    #         plt.subplot(3, 3, i + 1)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.grid(False)
    #         plt.imshow(image, cmap=plt.cm.binary)
    #         plt.xlabel(label)

    # plt.show()

    return train_ds, val_ds, class_names


def make_model(class_names):
    """Create a tf.keras.Model."""

    base_model = tf.keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=IMAGE_SHAPE
    )

    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    return model


def main():
    """Main function."""
    train_ds, val_ds, class_names = make_dataset()

    model = make_model(class_names)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
    )

    # model.save("model")

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
