# ASL MNIST Hand Sign Recognition

This project trains a CNN using TensorFlow and Keras to classify ASL hand signs from the Sign Language MNIST dataset (28×28 grayscale images). I achieved an F1 score of 0.99 on this dataset.

**Problems:**  
Real-world images (e.g., from a webcam) differ significantly from MNIST data—they're higher resolution, have complex backgrounds, and may include unwanted elements (like faces), which leads to poor real-time predictions despite the model's high performance on MNIST.

**Mediapipe/Transfer Learning:**  
To address these issues, I experimented with Google's Mediapipe landmark pipeline and transfer learning. I finetuned the Mediapipe gesture recognizer model to extract hand landmarks and adapt it for ASL recognition.  
[Google Colab Demo]([https://colab.research.google.com/drive/1qmt4F5M7FDTbBYy42dDhiIpQe5vTR22x#scrollTo=1_pux_SfseU5])

The resulting model now performs better on real-world images, though further finetuning is needed to improve its performance.
