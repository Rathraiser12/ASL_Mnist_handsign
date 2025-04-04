# ASL MNIST Hand Sign Recognition

This project trains a CNN using TensorFlow to classify ASL hand signs based on the Sign Language MNIST dataset (28×28 grayscale images). 

**Problems:**  
Real-world images (e.g., from a webcam) differ significantly from MNIST data—they're higher resolution, have complex backgrounds, and may include unwanted elements (like faces), leading to poor predictions.

**Mediapipe/Transfer Learning:**  
To address these issues, I experimented with Google's Mediapipe pipeline and transfer learning to extract hand landmarks, aiming to improve real-time hand sign detection.

