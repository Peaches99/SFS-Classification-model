import os
import time
import random
import numpy as np
import PIL
import psutil
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers


from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

IMAGE_SHAPE = (324, 324, 3)
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
DATA_DIR = "data/"
USE_CUDA = True


print("TensorFlow version: "+tf.__version__)
print("Pillow version: "+PIL.__version__)

process = psutil.Process(os.getpid())

if tf.test.is_built_with_cuda():
    print("CUDA is available")
    CUDA = True
    nvmlInit()
else:
    print("CUDA is NOT available")
    CUDA = False

# if use_cuda is set to False disable GPU
if USE_CUDA and CUDA:
    print("Using GPU")
else:
    print("Using CPU")
    CUDA = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    """Main function"""
    # measure the time
    start_time = time.time()

    rand_seed = random.randint(0, 1000)

    print("Loading images from "+DATA_DIR+" ...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, labels="inferred", label_mode="categorical", class_names=None,
        color_mode="rgb", batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        shuffle=True, seed=rand_seed,
        interpolation="bilinear",)

    class_names = dataset.class_names
    print("Class names: "+str(class_names))

    train_size = int(0.8 * len(dataset))

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    example_images = val_ds.take(9)

    print("Training dataset size: "+str(len(train_ds)))
    print("Validation dataset size: "+str(len(val_ds)))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    print("Loading complete after " +
          str(round(time.time()-start_time, 2))+" seconds")

    print("Applying transformations to the dataset...")
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
    dataset = dataset.map(lambda x, y: (
        tf.image.rot90(x, k=random.randint(0, 3)), y))
    dataset = dataset.map(lambda x, y: (
        tf.image.per_image_standardization(x), y))

    # create the model
    model = tf.keras.Sequential([
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(class_names))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # show current memory usage
    if CUDA:
        handle = nvmlDeviceGetHandleByIndex(0)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        print("GPU Memory usage: "+str(round(meminfo.used/1024/1024, 2)) +
              " MB/"+str(round(meminfo.total/1024/1024, 2))+" MB")
    else:
        print("Memory usage: "+str(round(process.memory_info().rss/1024/1024, 2)) +
              " MB/"+str(round(psutil.virtual_memory().total/1024/1024, 2))+" MB")

    print("Starting training...")
    start_time = time.time()

    # train the model
    model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/ant_model.h5',
                #save using the best validation accuracy
                monitor='val_accuracy',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                save_freq='epoch')
        ])

    # get the final validation accuracy
    print("Training complete after " +
          str(round(time.time()-start_time, 4))+" seconds")

    # load models/model
    final_model = tf.keras.models.load_model('models/ant_model.h5')
    val_acc = final_model.evaluate(val_ds)[1]
    # save the model with the validation accuracy in the name
    print("Final validation accuracy: "+str(round(val_acc*100, 2))+"%")

    # use the example images to make predictions
    print("Making predictions...")
    predictions = model.predict(example_images)
    print("Predictions complete")

    # show the predictions using matplotlib
    plt.figure(figsize=(10, 10))
    for images, labels in example_images:
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(predictions[i])])
            plt.axis("off")
    plt.show()
if __name__ == "__main__":
    main()
