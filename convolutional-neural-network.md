# An Intuitive Guide to Convolutional Neural Networks
source: https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050

## Convolutional Neural Networks

Computers "see"in a different way than we do. Their world consists of only numnbers. Every image can be represented as 2-dimensional arrays of numbers, known as pixels. But the fact that they perceive images in a different way, doesn't mean we can't train them to recognize patterns, like we do. We just have to think of what an image in a different way.

>![](https://cdn-images-1.medium.com/max/1600/1*ccVO7341XIh7GfvzQS1IGw.png)

To teach an algorithm how to recognize objects in images, we use a specific type of Artificial Neural Network: A Convolutional Neural Network (CNN). Their name stems from one of the most important operations in the network: convolution.

# A Beginner's Guide to Convolutional Neural Networks
source: https://skymind.ai/wiki/convolutional-network

## Introduction to Deep Convolutional Neural Networks
Convolutional Neural Networks are deep artificial neural networks that are used primarily to classify images (e.g. name what they see), cluster them by similarity (photo search), and perform object recognition within scenes. They are algorithms that can identify faces, individuals, street signs, tumors, platypuses and many other aspects of visual data.

Convolutional networks perform optical character recognition (OCR) to digitize text and make natural-language processing possible on analog and hand-written documents, where the images are symbols to be transcribed. CNNs can also be applied to sound when it is represented visually as a spectrogram. More recently, convolutional networks have been applied directly to text analytics as well as graph data with graph convolutional networks.

The efficacy of convolutional nets (ConvNets or CNNs) in image recognition is one of the main reasons why the world has woken up to the efficacy of deep learning. They are powering major advances in computer vision (CV), which has obvious applications for self-driving cars, robotics, drones, security, medical diagnoses, and treatments for the visually impaired.

## Images are 4D Tensors
Convolutional neural networks ingest and process images as tensors, and tensors are matrices of numbers with additional dimensions.

They can be hard to visualize, so let’s approach them by analogy. A scalar is just a number, such as 7; a vector is a list of numbers (e.g., [7,8,9]); and a matrix is a rectangular grid of numbers occupying several rows and columns like a spreadsheet. Geometrically, if a scalar is a zero-dimensional point, then a vector is a one-dimensional line, a matrix is a two-dimensional plane, a stack of matrices is a three-dimensional cube, and when each element of those matrices has a stack of feature maps atttached to it, you enter the fourth dimension. For reference, here’s a 2 x 2 matrix:

```python
[1, 2]
[5, 8]
```

The width and height of an image are easily understood. The depth is necessary because of how colors are encoded. Red-Green-Blue (RGB) encoding, for example, produces an image three layers deep. Each layer is called a “channel”, and through convolution it produces a stack of feature maps (explained below), which exist in the fourth dimension, just down the street from time itself. (Features are just details of images, like a line or curve, that convolutional networks create maps of.)

So instead of thinking of images as two-dimensional areas, in convolutional nets they are treated as four-dimensional volumes. These ideas will be explored more thoroughly below.

## Convolutional Definition
From the Latin *convolvere*, "to convolve" means to roll together. For mathematical purpose, a convolution is the integral measuring how much two functions overlap as one passes over the other. Think a convolution as a way of mixing two functions by multiplying them

## How Convolutional Neural Networks Work


# An Intuitive Guide to Convolutional Neural Networks 
https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050
## Architecture

Convolutional Neural Networks have a different architecture than regular Neural Networks. 

Regular Neural Networks transform an input by putting it through a series of hidden layers. Every layer is made up of a **set of neurons**, where each layer is fully connected to all neurons in the layer before. Finally, there is a last fully-connected layer - the output layer - that represent the predictions.

Convolutional Neural Networks are a bit different. First of all, the layers are organized in 3 dimensions: width, height, and depth. Further, the neurons in one layer do not connect to all the neurons in the next layer but only to a small region of it. Lastly, the final output will be reduced to a single vector of probability scores, organized along the depth dimension.

![](https://cdn-images-1.medium.com/max/1600/1*U8huw63urvRLUwJe89VXpA.png)

CNN have two components:
- The Hidden Layers / Feature Extraction part
    - In this part, the network will perform a series of **convolutions** and **pooling** operations during which the **features are detected**. If you had a picture of a zebra, this is the part where the network would recognizes its stripes, two ears and four legs.
- The Classification Part
    - Here, the fully connected layers will serve as a **classifier** on the top of these extracted features. They will assign a **probability** for the object on the image being what the algorithm predicts it is.
    
## Feature Extraction
Convolution is one of the main building blocks of a CNN. The term convolution refers to the mathematical combination of two functions to produce a third function. It merges two sets of information.

In the case of a CNN, the convolution is performed on the input data with the use of a **filter** or **kernel** (these terms are used interchangeably) to then produce a **feature map**.

We execute a convolution by sliding the filter over the input. At every location, a matrix multiplication is performed and sums the result onto the feature map.

![](https://cdn-images-1.medium.com/max/1600/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif)
In the animation above, you can see the convolution operation. You can see the **filter** (the green square) is sliding over our **input** (the blue squalre) and the sum of the convolution goes into the **feature map** (the red square). The area of our filter is also called the receptive field, named after the neuron cells! The size of this filter is 3x3.

For the sake of explaining, I have shown you the operation in 2D, but in reality convolutions are performed in 3D. Each image is namely represented as a 3D matrix with a dimension for width, height, and depth. Depth is a dimension because of the colours channels used in an image (RGB).

We perfom numerous convolutions on our input, where each operation uses a different filter. This results in different feature maps. In the end, we take all of these feature maps and put them together as the final output of the convolution layer.

Just like any other Neural Network, we use an **activation function** to make our output non-linear. In the case of a Convolutional Neural Network, the output of the convolution will be passed through the activation function. This could be the ReLU activation function.

**stride** is the size of the step the convolution filter moves each time. A stride size is usually 1, meaning the filter slides pixel by pixel. By increasing the stride size, your filter is sliding over the input with a larger interval and thus has less overlap between the cells.

Because the size of the feature map is always smaller than the input, we have to do something to prevent our feature map from shrinking. This is where we use **padding**.

After a convolution layer, it is common to add a pooling layer in between CNN layers. The function of pooling is to continuously reduce the dimensionality to reduce the number of parameters and computation in the network. This shortens the training time and controls overfitting.

The most frequent type of pooling is max pooling, which takes the maximum value in each window. These window sizes need to be specified beforehand. This decreases the feature map size while at the same time keeping the significant information.

