# Batch normalization: theory and how to use it with TensorFlow
source: https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad

Not so long ago, deep neural networks were really difficult to train, and making complex models converge in a reasonable amount of time would have been impossible. Nowadays, we have a lot of tricks to help them converge, to achieve faster training and to solve any kind of trouble that arise when we want to train a Deep Learning model. This article is going to explore one of those tricks: **batch normalization**.

## Why you should use it.
In order to understand what batch normalization is, first we need to address which problem it is trying to solve.

Usually, in order to train a neural network, we do some preprocessing to the input data. For example, we could normalize all data so that it resembles a normal distribution (that means, zero mean and a unitary variance). Why do we do this preprocessing? Well, there are many reasons for that, some of them being: preventing the early saturation of non-linear activation functions like the sigmoid function, assuring that all input data is in the same range of values, etc.

But the problem appears in the intermediate layers because the distribution of the activations is constantly changing during training. This slow down the training process because each layer must learn to adapt themselves to a new distribution in every training step. This problem is known as **internal covariate shift**.

So ... what happens if we force the input of every layer to have approximately the same distribution in every training step?

## What it is?
Batch Normalization is a method we can use to normalize the inputs of each layer, in order to fight the internal covariate shift problem.

During training time, a bach normalization layer does the following:
1. Calculate the mean and variance of the layers input.
![](https://cdn-images-1.medium.com/max/800/1*_6xWFQC0jb9_T1yz9iqOWA.png)

2. Normalize the layer inputs using the previously calculated batch statistics.
![](https://cdn-images-1.medium.com/max/800/1*I7YluVpp6-mfMoj4AZZI5g.png)

3. Scale and shift in orer to obtain the output of the layer. 
![](https://cdn-images-1.medium.com/max/800/1*G8-bO54pVT5eJCJ7MBabdA.png)

Notice that γ and β are learned during training along with the original parameters of the network.

So, if each batch had *m* samples and there were *j* batches:
![](https://cdn-images-1.medium.com/max/800/1*j9KW8tVE8XTEu6lcP1dFoQ.png)
Edit: During test (or inference) time, the mean and the variance are fixed. They are estimated using the previously calculated means and variances of each training batch.

## How do we use it in TensorFlow

Luckily for us, the Tensorflow API already has all this math implemented in the **tf.layers.batch_normalization** layer.

In order to add a batch normalization layer in your model, all you have to do is use the following code:
```python
import tensorflow as tf

# ...

is_train = tf.placeholder(tf.bool, name="is_train");

# ...

x_norm = tf.layers.batch_normalization(x, training=is_train)

# ...

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```

It is really important to get the update ops as stated in the Tensorflow documentation because in training time the moving variance and the moving mean of the layer have to be updated. If you don’t do this, batch normalization will not work and the network will not train as expected.

It is also useful to declare a placeholder to tell the network if it is in training time or inference time (we already discussed which are the differences for train and test time).

Notice that this layer has a lot more parameters (you can check them in the documentation), but these is the basic working code that you should use.
