---
layout: post
title: "A Movie Review Classification"
categories: tutorials
---

Moving on with the [Keras Tutorial](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb), today's article is a description of the movie review classifier using keras' imdb embedded dataset. 

# IMDB dataset

Similarly to MNIST, the IMDB dataset is embedded in keras, so it's very simple to import it into the notebook. 

```
import keras
from keras.dataset import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

It return two tuples: the training data and labels and the testing data and labels. The data is displaced as indexes of words. The most common word received index 1, the second most common word receives index 2, and so on. Index 0 does not stand for any specific word, but for encoding any unknown word. 

In the command above, we are filtering the data to contain only the 10000 most common words. So the dataset will never contain a word with index higer than 10000. This can be useful for filtering the most significative data. 

In train_data, there is an arra of 25000 reviews, and each review is an array of integers. In order to decode the review, one must use the dictionary index provided:

```
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
decode_review
```

In this case, 0, 1 and 2 are reserved indexes for padding, star of sequence and unknown. 


# Tutorial steps

In this section I cover what the tutorial is actually about. 

## Data preparation

The input data of our network cannot be only arrays of integers as presented in the previous section. There is the need to create layers and tensors to feed the neral network. 

For this, we first vectorize the data into arrays of 0s and 1s. 

```
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Vectoring labels as well
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```

## Building the Network

This was the most defiant part of the tutorial, since some concepts presented were not fully covered. First, our dataset is presented in the form of vectors of 1s and 0s, and that a etwork that has a good performance on it are the ones with the fully-conected layers with relu activation. 

Apart from that, the command that created the fully-connected layer (also called as Dense layer) receives a parameter of hidden units in the layer. This concept was briefly introduced as a dimension in the representation space of the layer. The tutorial mentions that these concepts will be explored in the future, but it's rather overwhelming not knowing why this network was designed with two intermediate layers with 16 units hidden each, and a third ouput layer with an scalar classification of the review's sentiment (negative or positive).

To actually build the network:

```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

## Compilation Step

```
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
```

### Optimization

At the compilation step, there are some extra functions to be defined:

* The optimization function
* The loss function
* And the metrix to monitor during the testing fase.

These previous steps are done by:

```
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])
```

## Training phase

The tutorial splitted the training data:

```
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```

And later we fit our model to 20 epochs:

```
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))
```

This tutorial presents fit monitoring with matplotlib.

The call to model.fit() returns a History object, which can be used to make this monitoring. This object contains a history dictionary, which contains data about everything that happened during training. 

```
history_dict = history.history
history_dict.keys()
```

The latest code line contains all the metrics being monitored during training. 

```
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

There was an analysis presented at the tutorial on the plots. It showed that *the training loss decreases with every epoch and the training accuracy increases with every epoch. That's what you would expect when running gradient descent optimization -- the quantity you are trying to minimize should get lower with every iteration. But that isn't the case for the validation loss and accuracy: they seem to peak at the fourth epoch.* 

Again, the subject of overfitting came to influence results. A possible solution to it was to train the model with 4 epochs intead of 20 (it showed a peak at the fourth epoch previously), but it only got 88% of accuracy. 

After that, we learn that it is possible to make predictions on data:

```
model.predict(x_test)
```

## Additional Experiments

TBD

## Conclusions

TBD

[Link](https://github.com/LucianaMarques/simpleCNN/blob/master/Notebooks/Movie%20Reviews%20Classifier.ipynb) to First Notebook.