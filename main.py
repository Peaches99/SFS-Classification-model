import os
import time
import sys
import numpy as np
import PIL
import tensorflow as tf
from sklearn.utils import shuffle

print("TensorFlow version: {}".format(tf.__version__))
print("Pillow version: {}".format(PIL.__version__))


def load(path):
    print("Loading data from {}".format(path)+" ...")
    images = []
    labels = []
    
    for folder in os.listdir(path):
        if folder == 'train' or folder == 'val':
            for subfolder in os.listdir(path+'/'+folder):
                if subfolder == 'ants' or subfolder == 'bees':
                    for image in os.listdir(path+'/'+folder+'/'+subfolder):
                        img = tf.keras.preprocessing.image.load_img(path+'/'+folder+'/'+subfolder+'/'+image, color_mode='rgb', target_size=(500,500))
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
    
    print("Train images: {}".format(train_images.shape), " Validation images: {}".format(val_images.shape))
    return train_images, train_labels, val_images, val_labels


images, labels = load('data/hymenoptera')

train_images, train_labels, val_images, val_labels = prepare(images, labels)

# Use vgg 16 as the base model for transfer learning

base_model = tf.keras.applications.vgg16.VGG16(input_shape=(500, 500, 3), include_top=False, weights='imagenet')

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#Make a prediction layer that consists of multiple layers

prediction_layer = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#history = model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))