import os

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from keras.applications import VGG19
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

data_dir = 'data/'
batch_size = 32
image_size = (224, 224)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)



base_model = VGG19(weights='imagenet', include_top=False, input_shape=(*image_size, 3))

# Add custom layers on top of the pretrained VGG19
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the
# base model (VGG19) for initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

def fgsm_adversarial_pattern(image, label, input_model, epsilon=0.01):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = input_model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adv_image = image + epsilon * signed_grad
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()

def generate_adversarial_images(generator, input_model):
    while True:
        x_batch, y_batch = next(generator)
        adv_batch = fgsm_adversarial_pattern(x_batch, y_batch, input_model)
        yield np.concatenate((x_batch, adv_batch)), np.concatenate((y_batch, y_batch))
        
        # Train the model with adversarial images
adversarial_train_generator = generate_adversarial_images(train_generator, model)
adversarial_val_generator = generate_adversarial_images(val_generator, model)

history = base_model.fit(
    adversarial_train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=adversarial_val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    class_weight=dict(enumerate(class_weights))
)

# ...

# Train the model for fine-tuning with adversarial images
fine_tuning_history = base_model.fit(
    adversarial_train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=adversarial_val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    class_weight=dict(enumerate(class_weights))
)
