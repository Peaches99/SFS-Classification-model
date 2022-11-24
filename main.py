import os
import time
import sys
from datetime import datetime
from packaging import version
import numpy as np
import PIL
from pynvml import *
import tensorflow as tf
from sklearn.utils import shuffle

threshold = 0.95

image_shape = (224,224,3)
epochs = 30
batch_size = 32
learning_rate = 0.0001

print("TensorFlow version: {}".format(tf.__version__))
print("Pillow version: {}".format(PIL.__version__))

if tf.test.is_built_with_cuda():
    print("CUDA is available")
    cuda = True
else:
    print("CUDA is NOT available")
    cuda = False
    
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load(path):
    print("Loading data from {}".format(path)+" ...")
    images = []
    labels = []
    
    for folder in os.listdir(path):
        if folder == 'train' or folder == 'val':
            for subfolder in os.listdir(path+'/'+folder):
                if subfolder == 'ants' or subfolder == 'bees':
                    for image in os.listdir(path+'/'+folder+'/'+subfolder):
                        img = tf.keras.preprocessing.image.load_img(path+'/'+folder+'/'+subfolder+'/'+image, color_mode='rgb', target_size=(image_shape[0], image_shape[1]))
                        images.append(tf.keras.preprocessing.image.img_to_array(img))
                        if subfolder == 'ants':
                            labels.append(0)
                        else:
                            labels.append(1)
    print("Images: {}".format(len(images)))
    return np.array(images), np.array(labels)


def prepare(images, labels):
    print("Preparing data ...")
    #convert the images to float32
    images = images.astype('float32')
    #normalize the images
    images /= 255
    #shuffle the images
    images, labels = shuffle(images, labels)
    #split the images into train and validation
    train_images, train_labels = images[:int(len(images)*0.8)], labels[:int(len(labels)*0.8)]
    val_images, val_labels = images[int(len(images)*0.8):], labels[int(len(labels)*0.8):]
   
    #turn everything into float
    train_images = train_images.astype('float32')
    val_images = val_images.astype('float32')
    train_labels = train_labels.astype('float32')
    val_labels = val_labels.astype('float32')
    
    print("Train images: {}".format(train_images.shape), " Validation images: {}".format(val_images.shape))
    return train_images, train_labels, val_images, val_labels


images, labels = load('data/hymenoptera')

train_images, train_labels, val_images, val_labels = prepare(images, labels)

model = tf.keras.models.Sequential()

#build the model
model.add(tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=image_shape))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.summary()

# make a callback early stopping that also prints the current memory usage and total time
class MemoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        if cuda:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.memory = nvmlDeviceGetMemoryInfo(self.handle).used
        self.start_time = time.time()
    
    def on_train_end(self, logs={}):
        if cuda:
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            end_memory = nvmlDeviceGetMemoryInfo(h).used
            nvmlShutdown()
            print("Memory at end: {} MB".format(end_memory//1024//1024))
            print("Total memory used: {} MB".format((end_memory-self.start_memory)//1024//1024))
            sys.stdout.flush()
        print("Total time: {} seconds".format(time.time()-self.start_time), flush=True)
    
    def on_epoch_end(self, epoch, logs={}):
        if cuda:
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            self.start_memory = nvmlDeviceGetMemoryInfo(h).used
            nvmlShutdown()
        self.epoch_start_time = time.time()
        # if the validation accuracy and the training accuracy are both above 0.99, stop training and save the model
        if logs.get('val_accuracy') > threshold and logs.get('accuracy') > threshold:
            print("Reached 99% accuracy, stopping training")
            self.model.stop_training = True
            self.model.save('models/ant_bee_model_{}.h5'.format(history.history['val_accuracy'][-1]))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[MemoryCallback()])

#save the model and put the accuracy in the name
model.save('models/ant_bee_model_{}.h5'.format(round(history.history['val_accuracy'][-1], 4)))

