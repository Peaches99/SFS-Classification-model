import os
import string
import time
import sys
import random
import numpy as np
import PIL
import psutil
import tensorflow as tf
import skimage
#import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers


from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from sklearn.utils import shuffle

THRESHOLD = 0.95
IMAGE_SHAPE = (224, 224, 3)
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DATA_DIR = "data/hymenoptera"
USE_CUDA = True

print("TensorFlow version: "+tf.__version__)
print("Pillow version: "+PIL.__version__)

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


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    """Main function"""
    # measure the time
    start_time = time.time()

    print("Loading images from "+DATA_DIR+" ...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, labels="inferred", label_mode="categorical", class_names=None,
        color_mode="rgb", batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        shuffle=True, seed=123, validation_split=0.2, subset="training",
        interpolation="bilinear",)

    print("Loading complete after " +
          str(round(time.time()-start_time, 2))+" seconds")
    # print the totoal images in training and validation
    # print the total image batches in training and validation
    print("Total Batches in training: "+str(dataset.cardinality().numpy()))
    print("Class names: "+str(dataset.class_names))

    print("Applying transformations to the dataset...")
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
    dataset = dataset.map(lambda x, y: (
        tf.image.rot90(x, k=random.randint(0, 3)), y))
    dataset = dataset.map(lambda x, y: (
        tf.image.per_image_standardization(x), y))

    print("Transformations applied")

    # use dataset cache and prefetch to improve performance
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # show current memory usage
    if CUDA:
        handle = nvmlDeviceGetHandleByIndex(0)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        print("GPU Memory usage: "+str(round(meminfo.used/1024/1024, 2)) +
              " MB/"+str(round(meminfo.total/1024/1024, 2))+" MB")
    else:
        print("Memory usage: "+str(round(process.memory_info().rss/1024/1024, 2)) +
              " MB/"+str(round(psutil.virtual_memory().total/1024/1024, 2))+" MB")


@tf.custom_gradient
def gradient_clipping(x):
    """ Clipping gradients to avoid exploding gradients """
    return x, lambda dy: tf.clip_by_norm(dy, 10.0)


def test_load(path):
    """ Loads the dataset from the given path """
    # get the images from the given path, resize them and run the model on them
    # then change the file names to either bee or ant or unknown
    print("Loading test images from "+path+" ...")
    load_images = []
    for image in os.listdir(path):
        img = tf.keras.preprocessing.image.load_img(
            path+'/'+image, color_mode='rgb', target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        load_images.append(tf.keras.preprocessing.image.img_to_array(img))
    return np.array(load_images)


if __name__ == "__main__":
    main()
