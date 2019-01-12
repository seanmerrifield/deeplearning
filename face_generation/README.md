# Face Generation using GANs

This project uses Generative Adverserial Networks (GANs) to generate a completely new set of faces that do not exist in reality. 

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset is used for this project, which contains over 200 thousand images of celebrity faces. 

![Celeb_Images](./images/celebrity_faces.png)

The results of the study are presented in the following sections, and the detailed execution of the code can be found in the accompanying Jupyter Notebook. 

## Installation

The installation instructions in the root of this repository should be used. No further installation is required. 

## GANs

A Generative Adverserial Network (GAN) is a semi-supervised learning model that is capable of generating a completely new set of data that didn't exist in reality, based on patterns that it had detected from data that it was trained on. It is a relatively recent model that was first [published](https://arxiv.org/abs/1406.2661) by Ian Goodfellow back in June 2014.

The principle idea behind GANs is that there are two networks, a generator and a discriminator, that are competing against each other:
- The generator makes fake data to pass to the discriminator
- The discriminator takes in the fake data from the generator and real data from the training set and has to decipher whether the data is real or fake. 

The generator is trained to fool the discriminator, such that the data it produces is as similar to the real data as possible. The discriminator is trained to decipher which data is real and which is fake. 

![GAN_Model](./images/gan_diagram.png)

GAN model architecture [credit Udacity.com]

## Model



## Training

