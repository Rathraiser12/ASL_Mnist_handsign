# ASL MNIST Hand Sign Recognition

This project trains a CNN using TensorFlow and Keras to classify ASL hand signs from the Sign Language MNIST dataset (28×28 grayscale images). I achieved an F1 score of 0.99 on this dataset.

The model uses 3 convolution blocks, with batch norm, L2 regularization, dropout and tanh activation function. I also use cuda to train locally on my gpu.

**Setbacks:**  
Real-world images (e.g., from a webcam) differ significantly from MNIST data—they're higher resolution, have complex backgrounds, and may include unwanted elements (like faces), which leads to poor real-time predictions despite the model's high performance on MNIST.

**Mediapipe/Transfer Learning:**  
To address these issues, I experimented with Google's Mediapipe landmark pipeline and transfer learning. I finetuned the Mediapipe gesture recognizer model to extract hand landmarks and adapt it for ASL recognition.  
[Link to Colab Model](https://colab.research.google.com/drive/1qmt4F5M7FDTbBYy42dDhiIpQe5vTR22x#scrollTo=1_pux_SfseU5)

The resulting model now performs better on real-world images, though further finetuning is needed to improve its performance.

**Third-Party Licenses**

This project uses [Google MediaPipe](https://mediapipe.dev/), which is licensed under the Apache License 2.0. Please refer to [MediaPipe's license](https://github.com/google/mediapipe/blob/master/LICENSE) for details.
