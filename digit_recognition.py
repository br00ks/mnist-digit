# authors: Karin Lampesberger, Maria Stoyanova

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import plot_after_preprocessing, plot_value_array, plot_prediction_image
# disables an annoying warning caused by existing gpu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get backend/output type
plt.get_backend()

# get mnist dataset
digits_mnist = keras.datasets.mnist

# training and test sets (data that uses model to learn)
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()
print("Dataset TRAINING images shape: {shape} ".format(shape=train_images.shape))
print("Dataset TESTING images shape: {shape} ".format(shape=test_images.shape))

# for plotting the images
class_names = ['0', '1', '2', '3', '4', 
               '5', '6', '7', '8', '9']

# preprocessing, both training images and testing images the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# plotting to see if images have the correct labels
plot_after_preprocessing(train_images, class_names, train_labels)

# build model and set up layers of neural network
# first keras layer flatten 28x28 array into 784 pixel vector
# second keras layer is fully connected 128 node layer with ReLu activation
# output layer is fully-connected 10 node softmax layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model (configure learning process)
# optimizer: defines how model is updated based on the data it sees and its loss function
# loss function: measures how accurate model is during training (function should be minimized)
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
# batch size (default: 32) = number of samples that are processed during training before an update
# of the model is done
# epochs = number of times training dataset will be passed through the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# compare how model performs on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Accuracy on test dataset:', test_acc)
print('Make predictions: ')

# make predictions
# each prediction is an array of 10 values
# max value = model's highest confidence classification
predictions = model.predict(test_images)
predictions[0]
print("The 1st sample in the test set is predicted to be: {pred}".format(pred=class_names[test_labels[0]]))
print("The 2nd sample in the test set is predicted to be: {pred}".format(pred=class_names[test_labels[1]]))
print("The 3rd sample in the test set is predicted to be: {pred}".format(pred=class_names[test_labels[2]]))
print("... and so on")

# plot first 12 test images with predicted and true label
# correct predictions are green, wrong predictions are red
num_rows = 4
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_prediction_image(i, predictions, test_labels, test_images, class_names)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# some more testing of the first image
img = test_images[0]
# add to batch, because tf.keras models are optimized so make predictions on batches
img = (np.expand_dims(img,0))
# print(img.shape)
predictions_single = model.predict(img)
# print(predictions_single)
plt.figure()
plt.grid(False)
plt.xticks([])
plt.yticks([])
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.savefig('test_prediction')
np.argmax(predictions_single[0])