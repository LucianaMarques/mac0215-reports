---
layout: post
title: "A First Look at Neural Networks"
categories: tutorials
---

This is an article about my work with the first tutorial on Deep Learning with Python Notebooks. 

What tutorial aims to teach is down to the basic usage of Keras and it presents the student the MNIST dataset. 

## MNIST Background and Origins

I thought it would be interesting to understand MNIST's background since it is very mentioned in articles and tutorials, so I did some quick research on it. The first thing to know is that Keras has a dataset module and it can be imported into your notebook with the following command:

```
import keras
from keras.dataset import <dataset_name>
```

In the following code sequence, one can replase `dataset_name` for any dataset that Keras provides, including MNIST. For reference, Keras has some [good documentation](https://keras.io/datasets/) about this module, so I discovered that we have the following at Keras' disposal:

* CIFAR10: small image classification labeled over 10 categories
* CIFAR100: small image classification labeled over 100 categories
* IMDB Movie reviews sentiment classification (this seemed particularly interesting)
* Reuters newswire topics classification (this seemed interesting as well)
* MNIST database of handwritten digits
* Fashion-MNIST database of fashion articles
* Boston housing price regression dataset 

The last one I remember using for projects of a previous class I took on supervised learning and intro to computer vision. 

Now, about MINIST itself, it started as handwritten digit database which was *"originally set and experimented by Chris Burges and Corinna Cortes using bounding-box normalization and centering. Yann LeCun's version which is provided on this page uses centering by center of mass within in a larger window."* accordingly to [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/index.html). I took some time to learn about professor Lecun and also Chris Burges and Corinna Cortes, and found very interesting to find a woman's collaboration on this. 

## Tutorial steps

In this section I cover what the tutorial is actually about. 

### Data visualization

A very import pictured in this tutorial is to understand how data is shaped. In case of MNIST, there are two different datasets, the training and the testing ones, which division seems to be quite common. The training datasets contain 60000 samples 28x28 matrixes.

### Building the Network

There are two import models in Keras for deeplearning. They are models and layers.

```
from keras import models
from keras import layers
```

The first one is to create the network, which is done by:

```
network = models.Sequential()
```

This means that the variable network will contain the neural network we are building. 

The next step is to add the layers that were previously designed in the model and then compile the network. 

At the compilation step, there are some extra functions to be defined:

* The optimization function
* The loss function
* And the metrix to monitor during the testing fase.

These previous steps are done by:

```
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
```

### Training phase

The first thing to do in this phase is to encode the labels:

```
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

On the tutorial, the reason for this step was not explained and it mentioned that this concept would be explored in a future chapter. I tried to understand it in advance and I figured out it would be because we need to 

The next step is probably the most important of the training phase, which is to fit the network to the model we created. 

```
network.fit(train_images, train_labels, epochs=5, batch_size=128)
```

In the notebook I generated, is is possible in this step to check the accuracy in each "epoch". The final accuray obtained for this fit was 0.9885

### Testing phase

The final fase is to test the model crated with the testing data. This basically will tell if our model is good or not. We get the testing accurary by:

```
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_acc)
```

The final accuracy was 0.9778, which is less from the training accuracy. According to the tutorial, this could indicate an overfitting case.

## Final Notebook, conclusion and next steps

This tutorial was followed to get the basics of Keras and to relate it to the knowledge gained by the deep learning reading material. It was very informative on the basics. 

A possible following step to this would be investigate why is this model producing an overfitting. And a certain next job is to continue with the subsequent tutorials.

The final notebook produced by me on this first tutorial can be accessed here: (link)[https://github.com/LucianaMarques/simpleCNN/blob/master/Notebooks/A%20first%20simple%20NN.ipynb]