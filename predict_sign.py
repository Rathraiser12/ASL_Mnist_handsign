import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define the remove_face function
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def remove_face(image):
    """
    Detects faces in the input image and masks them out.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Mask out the face region by drawing a filled rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return image

def load_and_preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found or cannot be loaded.")
        return None
    # Resize to 28x28 (as expected by the model)
    image = cv2.resize(image, (28, 28))
    # Normalize pixel values
    image = image.astype("float32") / 255.0
    # Add channel dimension and batch dimension
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return None

    print("Press 's' to capture an image or 'ESC' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        # Optionally, display instructions or an ROI overlay here
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Remove the face from the captured frame
            frame_no_face = remove_face(frame)
            gray = cv2.cvtColor(frame_no_face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (28, 28))
            gray = gray.astype("float32") / 255.0
            gray = np.expand_dims(gray, axis=-1)
            gray = np.expand_dims(gray, axis=0)
            cap.release()
            cv2.destroyAllWindows()
            return gray
        elif key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    # Load the best saved model
    model = tf.keras.models.load_model("best_model.h5")
    
    # Updated class mapping for the Sign Language MNIST dataset (24 classes)
    class_mapping = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
        5: "F", 6: "G", 7: "H", 8: "I", 10: "K",
        11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
        16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
        21: "V", 22: "W", 23: "X", 24: "Y"
    }
    
    choice = input("Enter '1' for Webcam capture or '2' for Image path: ").strip()
    if choice == "1":
        image_input = capture_from_webcam()
        if image_input is None:
            print("No image captured.")
            return
    elif choice == "2":
        image_path = input("Enter the image file path: ").strip()
        image_input = load_and_preprocess_image(image_path)
        if image_input is None:
            print("Image could not be processed.")
            return
    else:
        print("Invalid choice.")
        return
    
    plt.imshow(image_input.squeeze(), cmap="gray")
    plt.show()
    # Run the prediction
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_mapping.get(predicted_class, "Unknown")
    print(f"Predicted Sign: {predicted_label}")

if __name__ == "__main__":
    main()
