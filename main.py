from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model

import tensorflow as tf

if __name__ == "__main__":
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU detected. Using CUDA.")
        for gpu in gpus:
            print("GPU:", gpu)
        device = "/GPU:0"
    else:
        print("❌ No GPU detected. Running on CPU.")
        device = "/CPU:0"

    # Load and preprocess data
    train_df, test_df = load_data("data/sign_mnist_train.csv", "data/sign_mnist_test.csv")
    x_train, x_val, x_test, y_train, y_val, y_test, num_classes = preprocess_data(train_df, test_df)

    # Build, train, and evaluate the model on the selected device
    with tf.device(device):
        model = build_model((28, 28, 1), num_classes)
        model, history = train_model(model, x_train, y_train, x_val, y_val)
        evaluate_model(model, x_test, y_test, history)
