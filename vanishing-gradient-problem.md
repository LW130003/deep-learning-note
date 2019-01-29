# Vanishing Gradient Problem
source: https://en.wikipedia.org/wiki/Vanishing_gradient_problem

In machine learning, the **vanishing gradient problem** is a difficulty found in training artificial neural networks with **gradient-based leanring methods** and **backpropagation**.

In such methods, each of the neural network's weights receives an update proportional to the **partial derivative** of the **error function** with respect to the current weight in each iteration of training. The problem is that in some cases, the gradient will be vanishingly small, effectively preventing the weight from changing its value. In the worst case, this may completely stop the neural network from further training.

As one example of the problem cause, traditional **activation functions** such as the **hyperbolic tangent** function have gradient in the range (0,1), and backpropagation computes gradients by the chain rule. This has the effect of multiplying *n* of these small numbers to compute gradients of the "front" layers in an *n*-layer network, meaning that the gradient (error signal) decrease exponentially wiht *n* while the front layers train very slowly.


