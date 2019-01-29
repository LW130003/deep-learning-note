# Why does an internal covariate shift slow down the training procedure?
source: https://www.quora.com/Why-does-an-internal-covariate-shift-slow-down-the-training-procedure

Let's say you have a goal to reach, which is easier, a fixed goal vs a goal that keeps moving about? It is clear that a static goal is much easier to reach than a dynamic goal.

Each layer in a neural net has a simple goal, to model the input from the layer below it, so each layer tries to adapt to it's input but for hidden layers, things get a bit complicated. The input's statistical distribution changes after a few iterations, so if the input statistical distribution keeps changing, called **internal covariate shift**, the hidden layers will keep trying to adapt to that new distribution hence slowing down convergence. It is like a goal that keeps changing for hidden layers.

So the Batch Normalization (BN) algorithms tries to normalize the inputs to each hidden layer so that their distribution is fairly constant as training proceeds. This improves converggence of the neural net.

Looking at your comment: "I think it is because when data flows through the non-linear parts, it saturates as the parameters of previous layers changes and cause gradient vanishing."

The normalization also makes the neurons work in the linear regions of their activation functions further improving learning and recognition performance. BN also prevents vanishing gradient problem, so sigmoid and tanh can be used without much trouble.

## ...

Covariate shift describes what happens when your input distribution (the X_i) that goes into your model changes between what you trained the model on and what you test or deploy it on.

In your typical neural network, what you do is you scale your data to 0 mean and unit variance. Then you propagate the data through the hidden layers. At each hidden layer, you perform the usual linear Wx +b and then run it through the activation. Once you do this, the output A that goes into the next layer no longer has 0 mean and unit variance. This output A is like the new set of data that is the input for your next hidden layer (or submodel) to learn on. Since this new data no longer has unit variance and 0 mean like your input data, it’s as if you changed the distribution of your data, which is not good for your model.

Batch normalization addresses this problem for neural networks. Assume you feed in a batch or subset of the training data. At every layer, you always rescale the data to 0 mean and unit variance with respect to the samples.

