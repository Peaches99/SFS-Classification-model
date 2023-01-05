"""This script is used to train a model on the dataset in the data/ folder."""

import os
import random
import time
import PIL
import psutil
import tensorflow as tf


from pynvml import nvmlInit

IMAGE_SHAPE = (224, 224, 3)
EPOCHS = 50
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

example_images = val_ds.take(9)

print("Training dataset size: " + str(len(train_ds)))
print("Validation dataset size: " + str(len(val_ds)))

autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)

print("Loading complete after " + str(round(time.time() - start_time, 2)) + " seconds")
