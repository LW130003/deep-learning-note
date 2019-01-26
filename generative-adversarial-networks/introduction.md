# Generative Adversarial Networks (GANs)

Generative Adversarial Networks were first introduced by Goodfellow et al. in their 2014 paper, *Generative Adversarial Networks*
Ian Goodfellow et al. “Generative Adversarial Nets”. In: Advances in Neural Information
Processing Systems 27. Curran Associates, Inc., 2014, pages 2672–2680. URL: http://
papers . nips . cc / paper / 5423 - generative - adversarial - nets . pdf (cited on
pages 245, 257).

These networks can be used to generate synthetic (i.e. fake) images that are perceptually near identiracl to their ground-truth, authentic originals.

In order to generate synthetic images, we make use of *two* neural networks during training:
1. A **generator** that accepts an input vector of randomly generated noise and prouces an output "imitation" image that looks similar, if not identical to an authentic image.
2. A **discriminator** or **adversary** which attempts to determine if a given image is an "authentic" or "fake".

By training both of these networks at the same time, one giving feedback to the other, we can learn to generate syntethci images.

## What are GANs?
The quintessential explanation of GANs typically involves some variant of two people working in
collusion to forge a set of documents, replicate a piece of artwork, or print counterfeit money - the counterfeit money printers is my personal favorite. In this example, we have two people:
- Jack, the counterfeit printer (the generator)
- Jason, an employee of the U.S. Treasury (which is responsible for printing money in the United States) who specializes in detecting counterfeit money (the *discriminator*)

Jack and Jason have been childhood friends, both growing up without much money in the rough parts of Boston. After much hard work in school, Jason was awarded a college scholarship — Jack was not and over time started to turn towards illegal ways to make money. He wasn’t very good, but he knew he could get better with the proper training.

One day, after a few too many pints at a local pub during Thanksgiving holiday, Jason let it slip to Jack that he wasn’t happy with his job. He was underpaid. His boss was nasty and spiteful, often yelling and embarrassing Jason in front of other employees. Jason was even thinking of quitting. Jack saw an opportunity to use Jason’s access at the U.S. Treasury to create an elaborate counterfeit printing scheme. Their conspiracy worked like this:
1. Jack, the counterfeit printer, would print fake bills and then mix *both* the fake bills and real money together, then show them to the expert, Jason.
2. Jason would sort through the bills, classifying each bills as "fake" or "authentic", giving feedback to Jack along the way on how he can improve his counterfeiting printing.

At first, Jack is doing a pretty poor job at printing counterfeit money. But over time, with Jason’s guidance, Jack eventually improves to the point where Jason is no longer able to spot the difference between the bills. By the end of this process, both Jack and Jason have stacks of counterfeit money that can fool most people.

## The General Training Procedure

We've discussed what GANs are in terms of an analogy, but what is the actual *procedure* to train them? Most GANs are trained using a six step process.
1. We randomly generate a vector (i.e., noise). 
2. We pass this noise through our generator which generates an actual image.
3. We then sample authentic images from our training set and mix them with our synthetic images.
4. Train our discriminator using this mixed set. The goal of the discriminator is to correctly label each image as "real" or "fake".
5. We'll once again generate random noise, but this time we'll purposely label each noise vector as a "real image".
6. We'll then train GAN using the noise vectors and "real image" labels even though they are not actual real images.

The reason this process works is due to:
1. We have frozen the weights of the discriminator at this stage, implying that the discriminator is not learning when we update the weights of the generator.
2. We're trying to "fool" the discriminator into being unable to determine which images are real vs. synthetic. The feedback from the disciminator will allow the generator to learnn how to produce more authentic images.

## Guidelines and Best Practices when Training
GANs are notoriously hard to train due to an *evolving loss landscape*. At each iteration of our algorithm we are:
1. Generating random images and then training the discriminator to correctly distinguish the two.
2. Generating additional synthetic images, but this time purposely trying to fool the discriminator
3. Updating the weights of the generator based on the feedback of the discriminator, therby allowing us to generate more authentic images.

From this process you'll notice there are two losses we need to minimize: one loss for the discriminator and a second loss for the generator. And since the loss landscape of the generator can be changed based on the feedback from the discriminator we end up with a dynamic system.

Our goal is not to seek a minimum loss value but instead find some equilibrium between the two. The concept of finding an equilibrium may make sense on paper, but once you try to implement and train your own GANs you'll find this is a non-trivial process.

Radford et al. recommends the following architecture guidelines for more stable GANs:
- Replace any pooling layers with strided convolutions (we have seen this concept used in ResNet chapter 11).
- Use batch normalization in both the generator and the discriminator
- Remove fully-connected layers in deeper networks
- Use ReLU in the generator except for the final layer which will utilize *tanh*
- Use Leaky ReLU in the discriminator.

Francois Chollet then provide additional recommendations in his book:
1. Sample random vectors from a *normal distribution* (i.e., Gaussian distribution) rather than a *uniform distribution*.
2. Add dropout to the discriminator
3. Add noise to the class labels when training the discriminator
4. To reduce checkerboard pixel artifacts in the output image use a kernel size that is divisible by the stride when utilizing convolution or transposed convolution in both the generator and discriminator.
5. If your adversarial loss rises dramatically while your discriminator loss falls to zero, try reducing the learning rate of the discriminator and incresing the dropout of the discriminator.

