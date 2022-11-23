import os
import numpy as np
import PIL
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Pillow version: {}".format(PIL.__version__))

#load the images from data/hymenoptera.
#The images are organized in to a train and validation folder.
#Each folder has two subfolders, one for ants and one for bees.
#The ants and bees images are labeled 0 and 1 respectively.
#The images dont have the same size, so we need to resize them to 500x500.
#Use any libraries you want to load the images and resize them.

def load(path):
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
    
    return np.array(images), np.array(labels)


def prepare(images, labels):
    #convert the images to float32
    images = images.astype('float32')
    #normalize the images
    images /= 255
    #shuffle the images
    images, labels = tf.keras.utils.shuffle(images, labels)
    #split the images into train and validation
    train_images, train_labels = images[:int(len(images)*0.8)], labels[:int(len(labels)*0.8)]
    val_images, val_labels = images[int(len(images)*0.8):], labels[int(len(labels)*0.8):]
    
    return train_images, train_labels, val_images, val_labels