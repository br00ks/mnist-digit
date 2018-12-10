# Building a neural network to recognize handwritten digits
Simple neural network to solve the MNIST digit classification problem using Tensorflow.
We are using the famous MNIST dataset, which consists of pictures of handwritten digits from 0 to 9. Each picture is a 28x28 pixel grayscale image. The training dataset consists of 60.000 images, the testing dataset of 10.000 images.

## Preprocessing
Before we can start we need to normalise the pixel values, so that they lie between 0 and 1 instead of 0 and 255. In order to do so, we divide the training and test datasets by 255.

## Model
- 1st keras layer flattens 28x28 array into 784 pixel vector
- 2nd keras layer is  fully connected 128 node layer with ReLu activation
- Output layer is fully connected 10 node softmax layer

## References:
- Oh, Il-Seok, and Ching Y. Suen. "A class-modular feedforward neural network for handwriting recognition." pattern recognition 35.1 (2002): 229-244.
- Graves, Alex, et al. "Unconstrained on-line handwriting recognition with recurrent neural networks." Advances in neural information processing systems. 2008.
- Abadi M. et. al.: “TensorFlow: A System for Large-Scale Machine Learning”, Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation, 2016.
- Build and train models in TensorFlow. Online available on 3.12.2018 at:
https://www.tensorflow.org/tutorials/keras/basic_classification 
- Feed-Forward Neural Net for MNIST, Online available on 3.12.2018 at:
https://wpovell.github.io/posts/ffnn-mnist.html?fbclid=IwAR1mR7rsBR5vIlO2kElWAqf1Qq74UA4Z4HqcIpWOzYcfj8FMKkFZifOp8Ug#Hidden-Layer-&-ReLU
- LeCun, Y., Boser, B. E., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W. E., & Jackel, L. D. (1990). Handwritten digit recognition with a back-propagation network. In Advances in neural information processing systems (pp. 396-404).
http://papers.nips.cc/paper/293-handwritten-digit-recognition-with-a-back-propagation-network.pdf



