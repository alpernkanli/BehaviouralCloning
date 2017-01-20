# BehaviouralCloning
Self-Driving Car Behavioural Cloning Application

# Introduction

In this project, a behavioural cloning approach is applied for making a car completely autonomous. The controlling mechanism is mechanism implemented "end-to-end", totally deep neural network.

# Data and Data Augmentation

Data is the most important part of this project. Using proper training data differs much more than using different models. The main source of data which is used in this project is provided by Udacity. But the important thing was not having a large amount of original data, it was augmenting data properly to generalize the outcome.  

Various data augmentation methods are implemented. They are;  

1- Changing brightness ("value") of picture randomly:  
  This is for generalizing the model for different light conditions.  
2- Changing main color ("hue") of the picture randomly:  
  This is for generalizing the model for differently colored roads.  
(Why just not using S channel of an HSV image: The change in the "value" and "hue" are limited in my model. I did not want to generalize model on every picture with same "saturation" pattern.)  
3- Jittering the steering angles:  
  There is not one single particular special angle for any turn. The car could have a range of different steering angles for a particular turn. So, for more generalization, angles are randomly jittered.  
4- Flipping the images:  
  The turning data are not balanced. To nearly balance it, some of the images are randomly flipped.  
5- Cropping:  
  The images are cropped to exclude areas not defining the steering angles.  

Data is passed to the model in the training time by a generator, to avoid memory issues.  

# The Architecture of the Model

The architecture is a small convolutional neural network. Dropout is used to prevent overfitting. Adding regularization did not made a significant change.

Architecture is based on most common convolutional neural network pattern in image recognition problems. As in my traffic sign recognition projects, Conv -> Activation -> Pool pattern is repeated two times to extract necessary features (two lines actually), and using two dense layers before the output.

Subsampling is only made in pooling layers. As activation function, ELU is used to overcome "dying ReLU" problem. Using dropout in just after the first dense layer (with more weights!) was enough to prevent overfitting.

The images are resized to 32, and this made the network even smaller.

![alt tag](https://raw.githubusercontent.com/alpernkanli/BehaviouralCloning/master/model.png)


# Training

The training is made by using python generators.By using fit_generator function in Keras, it became easy. The batches of 100 images are generated from the original Udacity data with data augmentation techniques, and this is made 20 times per epoch. Training for 4 epochs was enough for a smooth driving.

# Experimental

There are also working models with 16x16 and 8x8 sized image inputs, and really small networks in the "Experimental" section.

# Future Plans

1- Make the network as small as possible  
2- Make the model as generalized as possible, try it in real images  
3- Make the car as fast responding as possible  
4- Make the predicted angle sequence as smooth as possible  
