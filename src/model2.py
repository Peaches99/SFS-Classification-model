import os

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from keras.applications import VGG19
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_dir = "data/"
batch_size = 32
image_size = (224, 224)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest',
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)


base_model = VGG19(weights="imagenet", include_top=False, input_shape=(*image_size, 3))

# Add custom layers on top of the pretrained VGG19
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)
predictions = Dense(len(train_generator.class_indices), activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# Freeze the layers of the
# base model (VGG19) for initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "models/best_model.h5", monitor="val_loss", save_best_only=True
)

# Train the model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stopping, model_checkpoint],
)

# Unfreeze some layers of the base model (VGG19) for fine-tuning
for layer in base_model.layers[-6:]:
    layer.trainable = True

# Compile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "models/best_fine_model.h5", monitor="val_loss", save_best_only=True
)

# Train the model for fine-tuning
fine_tuning_history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stopping, model_checkpoint],
)
