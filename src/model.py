from tensorflow.keras import models, layers, regularizers

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(
            32, (3, 3), 
            activation='tanh', 
            kernel_regularizer=regularizers.l2(0.001), 
            input_shape=input_shape
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64, (3, 3), 
            activation='tanh', 
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128, (3, 3), 
            activation='tanh', 
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(
            128, 
            activation='tanh', 
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
