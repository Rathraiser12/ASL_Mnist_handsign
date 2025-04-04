import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data(train_df, test_df):
    x_train_full = train_df.drop('label', axis=1).values
    y_train_full = train_df['label'].values

    x_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    x_train_full = x_train_full.reshape((-1, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255

    num_classes = np.max(y_train_full) + 1
    y_train_full = to_categorical(y_train_full, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test, num_classes
