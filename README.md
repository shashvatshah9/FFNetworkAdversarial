# Adversarial Training Effects on Forward Forward Network

## ECE-GY-9163: Machine Learning for Cyber-security
---

## Overview

Deep learning algorithms have been used to construct a model that performs tasks like classification, regression, and now image, speech, and text generation. Such models are very accurate in making predictions. This project aims to implement and evaluate one of such algorithms the Forward-Forward (FF) algorithm proposed by G. Hinton, a novel learning procedure for neural networks. The FF algorithm replaces the forward and backward passes of back-propagation with two forward passes using positive and negative data, offering potential simplification in learning and video processing without the need for activity storage or derivative propagation. The focus of this project is on the simple feed-forward supervised method of the FF algorithm, implemented using PyTorch. These algorithms are very crucial for safety-critical applications. Increasing adversarial attack challenges the safety of using such algorithms. In this project, we have tested vulnerabilities of the proposed neural network with an adversarial attack and how adversarial training affects the performance of the network. This research aims to provide valuable insights into the performance and potential advantages of the FF algorithm.

## Topics:
- **Forward Forward Network implementation**
- **Model training with MNIST dataset**
- **Model training with CIFAR dataset**
- **Adversarial Attack**
- **Adversarial training**

## Dataset Overview
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)
This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Run the code
```
python main.py
```

## Results

Forward Forward Network training on MNIST:

![Image](/images/MNIST.png) 

Forward Forward Network training on MNIST:

![Image](/images/CIFAR10.png)

Adversarial attack results:

![Image](/images/attack_epsilon_vs_accuracy_before_adversarial.png)

Adversarial samples:

![Image](/images/attack_examples.png)

Adversarial attack after Adversarial Training:

![Image](/images/attack_epsilon_vs_accuracy_after_adversarial.png)

Comparison:

![Image](/images/acc-eps-adversarial.png)

## Conclusion
While Generative Adversarial Networks (GANs) excel at generating visually appealing images, they struggle with mode collapse, where large regions of the image space remain unexplored. Additionally, their reliance on backpropagation for both the generative and discriminative models raises compatibility concerns with biological neurons. The Forward-Forward (FF) algorithm offers a compelling alternative. It can be seen as a special type of GAN where each hidden layer of the discriminator independently classifies the input as positive or negative, negating the need for backpropagation in learning the generative model. This model simply reuses the representations learned by the discriminator and needs only a linear transformation to generate outputs, again avoiding backpropagation.

While initially falling behind the ResNet model at low perturbation levels, the FF displays remarkable resilience as epsilon values climb. Its consistent outperformance at higher levels underscores its potential as a robust defense against increasingly severe adversarial threats. Furthermore, adversarial training demonstrates its ability to further enhance the FF’s resilience. While subtle improvements occur at low epsilon values, its consistent performance at higher levels reinforces its innate aptitude for handling strong perturbations. This aligns with the model’s inherent characteristics, as observed in the non-adversarial training experiments. It’s crucial to remember that our analyses focused specifically on adversarial scenarios, not general test accuracy. This targeted evaluation provides valuable insights into the FF’s adaptability and resilience under attack, highlighting its potential as a robust and promising alternative for tackling the growing challenge of adversarial threats.