import os
import PIL
import psutil
import tensorflow as tf


from pynvml import nvmlInit

IMAGE_SHAPE = (224, 224, 3)
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DATA_DIR = "data/"
USE_CUDA = True


print("TensorFlow version: " + tf.__version__)
print("Pillow version: " + PIL.__version__)

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


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
