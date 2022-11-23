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