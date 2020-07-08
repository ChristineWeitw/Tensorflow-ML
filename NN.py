import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
fashion_mnist = keras.datasets.fashion_mnist
## split into training & testing
(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data()

# EDA
train_images.shape # 60,000 images with 28*28 pixels
## look into the umg
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()


class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Data Preprocessing (really important step for NN model)
## Normalization (NN started with random, so it's better to normalize the distribution of ur data)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create the Model
## Sequential (from left to right)
nn_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
## Flatten - all the pixel into 724
## Dense - connect the nerons from previous layer to this 128 neurons' layer (current layer)
## output layer (chosse 10 neurons accourding to the number of)
## Softmax - make sure all the neurons' value added up to 1 ( each of them is between [0,1])