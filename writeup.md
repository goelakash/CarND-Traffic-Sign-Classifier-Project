# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dist]: ./distribution.png "Visualization"
[graynorm]: ./gray_and_norm.png "Grayscaling"
[samples]: ./samples.png "Samples"
[testimg1]: ./download_test_images/Priority%20road.png "Priority Road"
[testimg2]: ./download_test_images/Right-of-way%20at%20the%20next%20intersection.png "Right-of-way at the next intersection"
[testimg3]: ./download_test_images/Stop.png "Stop"
[testimg4]: ./download_test_images/Turn%20right%20ahead.png "Turn right ahead"
[testimg5]: ./download_test_images/Wild%20animals%20crossing.png "Wild animals crossing"
[p1]: ./predictions/p1.png
[p2]: ./predictions/p2.png
[p3]: ./predictions/p3.png
[p4]: ./predictions/p4.png
[p5]: ./predictions/p5.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/goelakash/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here's a bar chart showing the proportion of items per class

![alt text][dist]

Here's a sample image from each class:
![alt text][samples]

It can be seen that some classes have very little examples in the dataset compared to others.

5 most common classes (Train):

5 least common classes (Train):

5 most common classes (Valid):

5 least common classes (Valid):

5 most common classes (Test):

5 least common classes (Test):


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the dataset size is not so large and using color instead of grayscale image cn cause the network to overfit. Also, as we see from the grayscale image samples side by side the color ones, we don't see any similarity b/w grayscale images from different classes. So it makes sense to not use the color channels.

After this, I normalized the image data to make the network converge faster and reduce learning time.

Here is an example of an original image and its grayed and normalized version:

![alt text][graynorm]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 RGB image                             | 
| Convolution 1 - 3x3   | 1x1 stride, same padding, outputs 32x32x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 16x16x64                 |
| Convolution 2- 3x3    | 1x1 stride, same padding, outputs 16x16x32    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x128                  |
| Fully connected 1     | outputs 300                                   |
| Fully connected 2     | outputs 100                                   |
| Fully connected 3     | outputs 43 (no. of classes)                   |
| Softmax               | 43 classes                                    |
|                       |                                               |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Batch-size: 128
* Epochs: 20
* Optimizer: Adam
* Loss function: Cross-entropy
* Learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with the LeNet architecture and tweaked the number of feature maps in the convolutional layer to get a good test accuracy. The validation accuracy on that model was around 91%. To improve that, I simply added a grayscale transformation before normalizing the image, and the validation accuracy went upto 95%.

My final model results (last epoch) were:
* training set accuracy of 100% (heavily overfitting)
* validation set accuracy of 92.9%
* test set accuracy of 93.5%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][testimg1] 
![alt text][testimg2] 
![alt text][testimg3] 
![alt text][testimg4] 
![alt text][testimg5]

The last image may be difficult to classify because it has other objects such as part of another sign, a pole, etc in it as well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction            | 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection | Right-of-way at the next intersection                                      | 
| Priority road | Priority road                        |
| Stop                    | Stop                       |
| Wild animals crossing | Wild animals crossing        |
| Turn right ahead | Turn right ahead                |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model probababilities for each of the image outputs are as shown.

| Image                    |     Prediction            | 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection | ![alt text][p1] | 
| Priority road | ![alt text][p2] |
| Stop                    | ![alt text][p3] |
| Wild animals crossing | ![alt text][p4] |
| Turn right ahead | ![alt text][p5] |

This is surprsing to me as the turn right ahead sign seemed pretty clear. On the other hand, the image for wild-animals crossing was less clear as it was slightly tilted and had other kinds of noise.

