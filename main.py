import tensorflow as tf
import numpy as np
import os
#import the PIL package
import PIL



print("TensorFlow version: {}".format(tf.__version__))

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
    
    
load("data/hymenoptera")