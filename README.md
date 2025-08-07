# Simple CNN for Handwritten Digit Recognition (MNIST)

This project demonstrates how to build and train a simple Convolutional Neural Network (CNN) using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. It also shows how to predict the digit in an image you upload.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction on Custom Images](#prediction-on-custom-images)
- [How to Run](#how-to-run)
- [Notes](#notes)

---

## Project Overview

This notebook builds a simple CNN model for digit classification using the MNIST dataset, which consists of grayscale images of handwritten digits (0-9). After training the model, you can evaluate its performance on the test set and use it to predict the digit in your own uploaded image.

---

## Dataset

- **Source:** [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Description:** 70,000 grayscale images (28x28 pixels) of handwritten digits (0-9)
    - 60,000 training images
    - 10,000 test images

---

## Model Architecture

- **Convolutional Layer #1:** 32 filters, 3x3 kernel, ReLU activation
- **Max Pooling Layer #1:** 2x2 window
- **Convolutional Layer #2:** 64 filters, 3x3 kernel, ReLU activation
- **Max Pooling Layer #2:** 2x2 window
- **Flatten Layer:** Converts 2D feature maps to 1D vector
- **Dense Layer (Hidden):** 64 neurons, ReLU activation
- **Dense Layer (Output):** 10 neurons, Softmax activation (for 10 classes)

---

## Data Preprocessing

- **Normalization:** Pixel values scaled from [0, 255] to [0, 1]
- **Reshaping:** Images reshaped to (num_samples, 28, 28, 1) for Keras (grayscale, single channel)
- **Visualization:** Optionally, a few training images are visualized using matplotlib

---

## Training

- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Epochs:** 5
- **Batch Size:** 64

---

## Evaluation

- The model is evaluated on the test set.
- Test accuracy and loss are printed.

---

## Prediction on Custom Images

You can upload your own image, preprocess it, and use the trained model to predict the digit.

**Steps:**
1. Upload an image using `files.upload()` in Google Colab.
2. Open the image, convert to grayscale, and resize to 28x28 pixels.
3. Normalize pixel values to [0, 1].
4. Reshape the image to (1, 28, 28, 1).
5. Predict using `model.predict()`.
6. Use `np.argmax()` to get the predicted digit.

**Example code:**
```python
from PIL import Image
import numpy as np


img = Image.open("your_image.png").convert("L")
img = img.resize((28,28))
img_dizi = np.array(img)
img = img_dizi.astype('float32') / 255
img = img.reshape((1,28,28,1))



tahmin = model.predict(img)
tahmin = np.argmax(tahmin)
print("Tahmin edilen rakam: ",tahmin)
```

---

## How to Run

1. Open the notebook in Google Colab or any Jupyter environment with TensorFlow and Keras installed.
2. Run all cells to:
    - Build and compile the model
    - Train the model on MNIST
    - Evaluate the model on the test set
3. (Optional) Upload your own image and test prediction as shown in the example above.

---

## Notes

- The uploaded image should be a clear, centered, single digit on a white background for best results.
- The model is kept simple (only 2 convolutional layers) for educational purposes and fast training.
- For more accurate results, you can increase the number of epochs or try more complex architectures.

---

Project by: Ahmet Recayi Öztürk
