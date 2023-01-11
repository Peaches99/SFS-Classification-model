"""This script is used to train a model on the dataset in the data/ folder."""

import os
import random
import time
import PIL
import numpy as np
from matplotlib import pyplot as plt
import psutil
import tensorflow as tf
from pynvml import nvmlInit

IMAGE_SHAPE = (270, 270, 3)
EPOCHS = 10
BATCH_SIZE = 16
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

    # split the dataset into train, validation and test
    train_ds = dataset.take(int(len(dataset) * 0.7))
    val_ds = dataset.skip(int(len(dataset) * 0.7))
    val_ds = val_ds.take(int(len(val_ds) * 0.6))
    test_ds = dataset.skip(int(len(dataset) * 0.7))
    test_ds = test_ds.skip(int(len(test_ds) * 0.6))

    print("Training dataset size: " + str(len(train_ds)))
    print("Validation dataset size: " + str(len(val_ds)))
    print("Test dataset size: " + str(len(test_ds)))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    print(
        "Loading complete after " + str(round(time.time() - start_time, 2)) + " seconds"
    )

    # example_images = val_ds.take(9)
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

    return train_ds, val_ds, test_ds, class_names


def calculate_class_weights(dataset, class_names):
    """Calculate the class weights for the dataset."""
    class_weights = {}
    for i in range(len(class_names)):
        class_weights[i] = 0
    for images, labels in dataset:
        for label in labels:
            label = label.numpy()
            label = label.argmax()
            class_weights[label] += 1
    for i in range(len(class_names)):
        class_weights[i] = 1 / class_weights[i]

    return class_weights


def main():
    """Main function."""
    train_ds, val_ds, test_ds, class_names = make_dataset()

    class_weights = calculate_class_weights(train_ds, class_names)

    # preprocess the dataset for vgg19
    preprocess_input = tf.keras.applications.vgg19.preprocess_input

    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    # make a custom callback that saves the best model and replaces it if a better one appears
    # dont actually save the best model as a file but only save it at the end

    class SaveBestModel(tf.keras.callbacks.Callback):
        def __init__(self):
            self.best_val_acc = 0
            self.best_model = None

        def on_epoch_end(self, epoch, logs=None):
            if logs["val_accuracy"] > self.best_val_acc:
                self.best_val_acc = logs["val_accuracy"]
                self.best_model = self.model

        def on_train_end(self, logs=None):
            self.model = self.best_model

    save_best_model = SaveBestModel()

    base_model = tf.keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=IMAGE_SHAPE
    )

    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="softmax"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[save_best_model],
    )

    model = save_best_model.best_model

    # train a second time with the base model trainable
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[save_best_model],
    )

    model = save_best_model.best_model

    evaluated = model.evaluate(test_ds, verbose=2)
    test_acc = evaluated[1]
    print("\nTest accuracy:", test_acc)

    model.save("./models/" + "sfs_model_" + str(round(test_acc, 2)) + ".h5")

    # plt.plot(history.history["accuracy"], label="accuracy")
    # plt.plot(history.history["val_accuracy"], label="val_accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.ylim([0.5, 1])
    # plt.legend(loc="lower right")
    # plt.show()

    # example_images = test_ds.take(18)
    # for images, labels in example_images:
    #     for i in range(18):
    #         image = images[i].numpy().astype("uint8")
    #         plt.subplot(3, 6, i + 1)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.grid(False)
    #         plt.imshow(image, cmap=plt.cm.binary)

    #         # predict
    #         prediction = model.predict(np.array([image]))

    #         # Get the probability in percent and put it next to the predicted label
    #         probability = prediction[0].max() * 100
    #         prediction = prediction[0].argmax()
    #         prediction = class_names[prediction]
    #         plt.xlabel(prediction + " (" + str(round(probability, 2)) + "%)")

    # plt.show()


if __name__ == "__main__":
    main()
