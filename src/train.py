from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Nadam

def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=64):
    # Compile the model
    model.compile(optimizer=Nadam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
         rotation_range=10,
         width_shift_range=0.1,
         height_shift_range=0.1,
         zoom_range=0.1
    )
    datagen.fit(x_train)
    
    # Set up callbacks: save best model based on validation loss and early stopping
    checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
    return model, history
