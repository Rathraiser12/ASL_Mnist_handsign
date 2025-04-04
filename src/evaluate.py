import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

def evaluate_model(model, x_test, y_test, history):
    # Evaluate test performance
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Predict on test set and calculate F1 score
    y_pred = model.predict(x_test)
    # Convert one-hot encoded predictions and true labels to class indices
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")
    
    # Plot training and validation loss from the history
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    return test_loss, test_acc, f1
