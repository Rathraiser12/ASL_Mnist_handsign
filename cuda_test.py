import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected. Using CUDA.")
    for gpu in gpus:
        print("GPU:", gpu)
else:
    print("No GPU detected. Running on CPU.")
