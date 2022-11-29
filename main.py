import os
import time
import sys
import random
import numpy as np
import PIL
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from sklearn.utils import shuffle

THRESHOLD = 0.95
IMAGE_SHAPE = (224, 224, 3)
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # best current results with 0.0001

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
        """initializes the callback"""
        self.plateau_threshhold = 10
        self.plateau_count = 0
        self.last_acc = 0.00
        self.early_end = False
        self.start_memory = 0
        self.device_handle = None
        self.start_time = time.time()
        self.best_model = None

    def on_train_begin(self, logs=None):
        """starts the timer and gets the device handle"""
        if CUDA:
            self.device_handle = nvmlDeviceGetHandleByIndex(0)

    def on_train_end(self, logs=None):
        """prints the time and memory used"""
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
        # save the best model
        self.best_model.save("models/model.h5")
        print("Model Accuracy: " +
              str(self.best_model.history.history['val_accuracy'][-1]))
        print(f"Total time: {elapsed} seconds")
        print(f"Average time per epoch: {elapsed/EPOCHS} seconds")

    def on_epoch_end(self, epoch, logs=None):
        """checks the accuracy and memory usage"""
        if CUDA:
            nvmlInit()
            self.start_memory = nvmlDeviceGetMemoryInfo(
                self.device_handle).used
            # print the current during the last epoch
        # safe the current best model
        if logs["val_accuracy"] > self.last_acc:
            self.last_acc = logs["val_accuracy"]
            self.best_model = self.model
            nvmlShutdown()

        current_acc = round(logs.get('val_accuracy'), 3)

        if current_acc == self.last_acc:
            self.plateau_count += 1

        if logs.get('val_accuracy') > THRESHOLD and logs.get('accuracy') > THRESHOLD or self.plateau_count == self.plateau_threshhold:
            self.early_end = True
            self.model.stop_training = True

        self.last_acc = round(logs.get('val_accuracy'), 3)

    def on_epoch_begin(self, epoch, logs=None):
        if CUDA:
            print(f"Memory usage: {(self.start_memory//1024//1024)} MB")


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


def prepare(images, labels):
    """Prepares the dataset for training"""

    images = images.astype('float32')
    labels = labels.astype('float32')

    # shuffle the data
    images, labels = shuffle(images, labels)

    # normalize the data
    images /= 255.0

    # plt.figure(figsize=(10, 10))
    # for i, image in enumerate(images[:9]):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(image)
    #     plt.title(int(labels[i]))
    #     plt.axis("off")
    # plt.show()

    # split the data into training and validation with a 80/20 split
    split = int(len(images)*0.8)
    train_images = images[:split]
    train_labels = labels[:split]
    val_images = images[split:]
    val_labels = labels[split:]

    return train_images, train_labels, val_images, val_labels


def train_single():
    """ trains the model on the dataset"""

    # load the data
    images, labels = load("data/hymenoptera")
    train_images, train_labels, val_images, val_labels = prepare(
        images, labels)


    # make a basemodel using resnet50
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=IMAGE_SHAPE)

    # freeze the base model
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SHAPE)

    model = base_model(inputs, training=False)
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(512, activation='relu')(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(1, activation='sigmoid')(model)
    model = keras.Model(inputs, model)

    model.summary()
    
    #train the top layer
    model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
    )

    model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(
        val_images, val_labels))


def main():
    """Main function"""
    train_single()
    # test_classify("test_images/")


@tf.custom_gradient
def gradient_clipping(x):
    """ Clipping gradients to avoid exploding gradients """
    return x, lambda dy: tf.clip_by_norm(dy, 10.0)


def test_load(path):
    # get the images from the given path, resize them and run the model on them
    # then change the file names to either bee or ant or unknown
    print("Loading test images from "+path+" ...")
    load_images = []
    for image in os.listdir(path):
        img = tf.keras.preprocessing.image.load_img(
            path+'/'+image, color_mode='rgb', target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        load_images.append(tf.keras.preprocessing.image.img_to_array(img))
    return np.array(load_images)


def test_classify(path):
    """Tests the model on the images in the given path"""
    print("Testing model on images in "+path+" ...")

    model = tf.keras.models.load_model('models/model.h5')
    # rename each image to a random number
    for index, image in enumerate(os.listdir(path)):
        os.rename(path+image, path+str(index)+'.jpg')
    loaded_images = test_load(path)
    # get the predictions
    predictions = model.predict(loaded_images)
    # rename each image to either bee or ant
    for i in range(len(predictions)):
        random_number = str(random.randint(0, 100000))
        print(predictions[i][0], predictions[i][1])
        if predictions[i][0] > predictions[i][1]:
            os.rename(path+str(i)+'.jpg', path+'ant'+random_number+'.jpg')
        else:
            os.rename(path+str(i)+'.jpg', path+'bee'+random_number+'.jpg')


if __name__ == "__main__":
    main()
