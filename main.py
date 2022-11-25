import os
import time
import sys
import numpy as np
import PIL
import tensorflow as tf

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from sklearn.utils import shuffle

THRESHOLD = 0.95
IMAGE_SHAPE = (300, 300, 3)
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 #best current results with 0.0001

print("TensorFlow version: "+tf.__version__)
print("Pillow version: "+PIL.__version__)

if tf.test.is_built_with_cuda():
    print("CUDA is available")
    CUDA = True
    nvmlInit()
else:
    print("CUDA is NOT available")
    CUDA = False


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MemoryCallback(tf.keras.callbacks.Callback):
    """A callback that stops the model training when the validation accuracy reaches a certain threshhold"""
    
    def __init__(self): 
        self.plateau_threshhold = 5
        self.plateau_count = 0
        self.last_acc = 0.00
        self.early_end = False
        self.start_memory = 0
        self.device_handle = None
        self.start_time = time.time()

    def on_train_begin(self, logs=None):
        if CUDA:
            self.device_handle = nvmlDeviceGetHandleByIndex(0)

    def on_train_end(self, logs=None):
        if CUDA:
            end_memory = nvmlDeviceGetMemoryInfo(self.device_handle).used
            print(f"Memory at end: {end_memory//1024//1024} MB")
            used_memory = (end_memory-self.start_memory)//1024//1024
            print(f"Total memory used: {used_memory} MB")
            sys.stdout.flush()
            nvmlShutdown()
        if self.early_end:
            print("\n\nReached Threshhold accuracy or plateaued, stopping training\n\n")
        elapsed = time.time()-self.start_time
        print(f"Total time: {elapsed} seconds")

    def on_epoch_end(self, epoch, logs=None):
        if CUDA:
            nvmlInit()
            self.start_memory = nvmlDeviceGetMemoryInfo(
                self.device_handle).used
            nvmlShutdown()

        current_acc = round(logs.get('val_accuracy'), 3)

        if current_acc == self.last_acc:
            self.plateau_count += 1

        if logs.get('val_accuracy') > THRESHOLD and logs.get('accuracy') > THRESHOLD or self.plateau_count == self.plateau_threshhold:
            self.early_end = True
            self.model.stop_training = True

        self.last_acc = round(logs.get('val_accuracy'), 3)


def load(path):
    """Loads the dataset from the given path"""

    print("Loading data from "+path+" ...")
    load_images = []
    load_labels = []

    for folder in os.listdir(path):
        if folder == 'train' or folder == 'val':
            for subfolder in os.listdir(path+'/'+folder):
                if subfolder == 'ants' or subfolder == 'bees':
                    for image in os.listdir(path+'/'+folder+'/'+subfolder):
                        img = tf.keras.preprocessing.image.load_img(
                            path+'/'+folder+'/'+subfolder+'/'+image, color_mode='rgb', target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
                        load_images.append(
                            tf.keras.preprocessing.image.img_to_array(img))
                        if subfolder == 'ants':
                            load_labels.append(0)
                        else:
                            load_labels.append(1)
    print("Images: "+str(len(load_images)))
    return np.array(load_images), np.array(load_labels)


def prepare(loaded_images, loaded_labels):
    """Prepares the dataset for training"""

    print("Preparing data ...")
    # convert the images to float32
    loaded_images = loaded_images.astype('float64')
    loaded_labels = loaded_labels.astype('float64')
    # normalize the images
    loaded_images /= 255
    # shuffle the images
    loaded_images, loaded_labels = shuffle(loaded_images, loaded_labels)
    # split the images into train and validation
    ptrain_images, ptrain_labels = loaded_images[:int(
        len(loaded_images)*0.8)], loaded_labels[:int(len(loaded_labels)*0.8)]
    pval_images, pval_labels = loaded_images[int(
        len(loaded_images)*0.8):], loaded_labels[int(len(loaded_labels)*0.8):]

    print(f"Train images: {ptrain_images.shape}",
          f" Validation images: {pval_images.shape}")
    return ptrain_images, ptrain_labels, pval_images, pval_labels


def main():
    """Main function"""
    train_single()


def train_single():
    """Trains a single model using the global variables"""

    images, labels = load('data/hymenoptera')

    train_images, train_labels, val_images, val_labels = prepare(
        images, labels)

    model = tf.keras.models.Sequential()

    # build the model
    model.add(tf.keras.applications.VGG16(include_top=False,
                                          weights='imagenet', input_shape=IMAGE_SHAPE))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callback = MemoryCallback()

    history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(val_images, val_labels), callbacks=callback)

    # save the model and put the accuracy in the name

    res = IMAGE_SHAPE[0]
    acc = round(history.history['val_accuracy'][-1], 4)
    model.save(f"models/ant_bee_{res}px_model_{acc}.h5")


if __name__ == "__main__":
    main()
