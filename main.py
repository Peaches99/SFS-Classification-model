import os
import time
import sys
import psutil
import numpy as np
import PIL
from pynvml import *
import tensorflow as tf
from sklearn.utils import shuffle

epochs = 50
batch_size = 16


print("TensorFlow version: {}".format(tf.__version__))
print("Pillow version: {}".format(PIL.__version__))

if tf.test.is_built_with_cuda():
    print("CUDA is available")
else:
    print("CUDA is NOT available")
    
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


base_model = tf.keras.applications.vgg16.VGG16(input_shape=(500, 500, 3), include_top=False, weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

#model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()

#history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), batch_size=batch_size)

# run multiple models to optimize the hyperparameters

#hyperparameters
learning_rate = [0.0001, 0.0002, 0.0005 , 0.001, 0.003, 0.005, 0.01, 0.1]
epochs = [50]
batch_size = [8, 16]
optimizer = ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
loss = ['binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'poisson', 'cosine_similarity']

#run the models
for lr in learning_rate:
    for e in epochs:
        for b in batch_size:
            for o in optimizer:
                for l in loss:
                    if o == 'adam':
                        opt = tf.keras.optimizers.Adam(learning_rate=lr)
                    elif o == 'rmsprop':
                        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
                    elif o == 'sgd':
                        opt = tf.keras.optimizers.SGD(learning_rate=lr)
                    elif o == 'adagrad':
                        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
                    elif o == 'adadelta':
                        opt = tf.keras.optimizers.Adadelta(learning_rate=lr)
                    elif o == 'adamax':
                        opt = tf.keras.optimizers.Adamax(learning_rate=lr)
                    elif o == 'nadam':
                        opt = tf.keras.optimizers.Nadam(learning_rate=lr)
                    else:
                        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
                    #compile the model
                    model.compile(optimizer=opt, loss=l, metrics=['accuracy'])
                    #fit the model
                    history = model.fit(train_images, train_labels, epochs=e, validation_data=(val_images, val_labels), batch_size=b)
                    #save the model if the accuracy is greater than 80%
                    if history.history['accuracy'][-1] > 0.8:
                        print("Model saved")
                        model.save('models/model_{}_{}_{}_{}_{}_{}'.format(lr, e, b, o, l, history.history['accuracy'][-1]))
                    #clear the session
                    tf.keras.backend.clear_session()
                    #print the results
                    print("Learning rate: {}".format(lr), " Epochs: {}".format(e), " Batch size: {}".format(b), " Optimizer: {}".format(o), " Loss: {}".format(l))
                    #print the results
                    print("Training accuracy: {}".format(history.history['accuracy'][-1]), " Validation accuracy: {}".format(history.history['val_accuracy'][-1]))
                    print("Training loss: {}".format(history.history['loss'][-1]), " Validation loss: {}".format(history.history['val_loss'][-1]))
