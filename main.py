import os
import time
import sys
import random
import numpy as np
import PIL
import tensorflow as tf

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


def prepare(loaded_images, loaded_labels):
    """Prepares the dataset for training"""

    loaded_images = loaded_images.astype('float64')
    loaded_labels = loaded_labels.astype('float64')

    print("Preparing data ...")
    # add noise to the images
    for i in range(len(loaded_images)):
        loaded_images[i] = tf.keras.preprocessing.image.random_shift(
            loaded_images[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        loaded_images[i] = tf.keras.preprocessing.image.random_rotation(
            loaded_images[i], 40, row_axis=0, col_axis=1, channel_axis=2)
        loaded_images[i] = tf.keras.preprocessing.image.random_zoom(
            loaded_images[i], (0.8, 1.2), row_axis=0, col_axis=1, channel_axis=2)

    # normalize the images
    loaded_images /= 255
    # shuffle the images
    loaded_images, loaded_labels = shuffle(loaded_images, loaded_labels)

    # prepare the data for usage with resnet50
    loaded_images = tf.keras.applications.resnet50.preprocess_input(
        loaded_images)

    # check if data has nan
    if np.isnan(loaded_images).any() or np.isnan(loaded_labels).any():
        print("Data has nan")
        sys.exit(1)
    
    # check if the data has inf or -inf as well as generally negative values
    if np.isinf(loaded_images).any() or np.isinf(loaded_labels).any() or np.min(loaded_images) < 0 or np.min(loaded_labels) < 0:
        print("Data has inf or -inf")
        sys.exit(1)

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
    test_classify("test_images/")


def train_single(optimizer='adam', learning_rate=0.0001,
                 momentum=0.9, loss='sparse_categorical_crossentropy'):
    """Trains a single model using the global variables"""

    optimizer = choose_optimizer(optimizer, learning_rate, momentum)

    images, labels = load('data/hymenoptera')

    train_images, train_labels, val_images, val_labels = prepare(
        images, labels)

    model = tf.keras.models.Sequential([
        tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_shape=IMAGE_SHAPE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # add a tiny bit of noise to the weights
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel = tf.keras.backend.random_normal(
                layer.kernel.shape, stddev=0.01)
            layer.bias = tf.keras.backend.random_normal(
                layer.bias.shape, stddev=0.01)
    # add a tiny number to the output of the last layer
    model.layers[-1].bias = tf.keras.backend.random_normal(
        model.layers[-1].bias.shape, stddev=0.0001)
    
    model.summary()
    
    # the loss returns nan as the value which is a problem for the optimizer so use the custom gradient_clipping
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    

    callback = MemoryCallback()

    history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(val_images, val_labels), callbacks=callback)

    # model.save(f"models/ant_bee_{IMAGE_SHAPE[0]}px_model_{test_acc}.h5")

    return model, history

@tf.custom_gradient
def gradient_clipping(x):
    """ Clipping gradients to avoid exploding gradients """
    return x, lambda dy: tf.clip_by_norm(dy, 10.0)

def choose_optimizer(optimizer, learning_rate, momentum):
    """Chooses an optimizer based on the given string"""
    optim = None
    if optimizer == 'adam':
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optim = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        optim = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        optim = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'adamax':
        optim = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == 'nadam':
        optim = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == 'ftrl':
        optim = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif optimizer == 'sgd_momentum':
        optim = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'sgd_nesterov':
        optim = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=True)
    else:
        print("Invalid optimizer, using Adam")
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optim


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
