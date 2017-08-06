## Project: Build a Traffic Sign Recognition Program
#### Writeup

###### **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram1.png "Training Distributions"
[image2]: ./examples/histogram2.png "Augmented distributions"
[image3]: ./examples/preprocessing_example.png "Preprocessing example"
[image4]: ./examples/data_aug.png "Data augmentation"

[image5]: ./examples/imgs_from_web.png "Images from web"

[image6]: ./examples/prediction_web.png "Prediction for web images"

### Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### 1. Provide a Writeup / README 

#### Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and numpy libraries to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (34799, 32, 32, 3)
Label data shape = (34799,)
Number of classes = 43

| Top 5 Signs: 		|SignCount  	  fraction of training set 		| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h) 	|	2010 	0.057760							| 
| Speed limit (30km/h) 	|	1980 	0.056898							|
|	 Yield				|	1920 	0.055174							|
| Priority road 		|	1890 	0.054312							|
| Keep right 	 	 	|	1860 	0.053450							|



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the unique data classes are distributed.

![alt text][image1]

here is a visualizarion of the augmented data set:
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Before preprocessing the dataset, I generated additional data for the unbalanced classes in the dataset. In the previous section you can appreciate the before and after distributions of the image classes. The data agumentation in my project was inspired by posts from the web (medium and forums) and it seemed as a very straightforward way to work with the already available data.

The data augmentation works this way:
1)With the function create variant, create a variation of the input image by randomly shifting or tilting it.
2)I Established a desired number of  observations per class and calculated the augmentation multiplier by dividing the desired number of observations between the actual number of observations per class.
3)Augment the data randomly  times the multiplier.

For the preprocessing of the augmented dataset:

As a first step, I decided to convert the images to grayscale because the 3 color channels may cause learning difficulties for the model and add complexity. Then I applied the equalizeHist function from cv2 library to make the images more bright. At last, I normalized the images.

Here is an example of a traffic sign image before and after grayscaling and after histogram equalizing:

![alt text][image3]


Here is an example of an original image and an augmented image:

![alt text][image4]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		|32x32x1 b&w image								| 
| Convolution 2D       	|1x1 stride, valid padding, outputs 28x28x28 	|
| RELU					|												|
| Max pooling			|1x1 stride, same padding  outputs 28x28x28		|
| Convolution 2D       	|1x1 stride, valid padding outputs 24x24x16		|
| RELU					|												|
| Max pooling			|1x1 stride, outputs 24x24x16, padding Same		|
| Conv2D				|2x2 stride, outputs 10x10x10, padding Valid	|
| RELU					|												|
| Max pooling			|1x1 stride, outputs 10x10x10, padding Same		|
| Flatten 				|size = 1000									|
| Fully connected		|size = 512, RELU								|
| Dropout				|Keep prob =  0.85								|
| Fully connected		|size = 256, RELU								|
| Dropout				|Keep prob =  0.85								|
| Fully connected		|size = 128, RELU								|
| Dropout				|Keep prob =  0.85								|
| Fully connected		|size = 64, RELU								|
| Dropout				|Keep prob =  0.85								|
| Fully connected  (out)|size = 43 classes, Linear						|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
Optimizer = Adam
Batch size = 512
Epochs = 15
Learning Rate = 0.00003

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9932
* validation set accuracy of 0.9707
* test set accuracy of 0.9409

* What was the first architecture that was tried and why was it chosen?
  The First arch that I tried consisted of: 2 conv2D, 2 maxpool, and then 3 fully connected layers because that worked
  another nanodegree project I made in the past.
  
* What were some problems with the initial architecture?
  The accuracy requisite was not met, not even closely.
  
* How was the architecture adjusted and why was it adjusted? 
  I adjusted the arch by adding more conv2D layers and then adding some dropout layers to avoid overfitting.

* Which parameters were tuned? How were they adjusted and why?
  The keep prob parameter first was set to 0.60 but that value was too low.
  Then I tuned the learning rate
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image5]



Some of the images are quite standard and other can be a challenge for the network to classify. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

8 out of 9 images were correctly classified!

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Turn right ahead 		| Turn right ahead  							|
| No entry				| Children Crossing								|
| Ahead only      		| Ahead only					 				|
| Keep right			| Keep right      								|
| Road work				| Road work      								|
| Yield					| Yield      									|
| No entry				| No entry      								|
| Ahead only			| Ahead only      								|

![alt text][image6]

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88%. This compares favorably to the accuracy on the test set of 0.94.

#### 3. Top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model is sure that this is a stop sign.

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 1    					| Stop sign   									| 
| 0    					| No entry 										|
| 0						| Go straight or left							|
| 0	      				| Speed limit (120km/h)			 				|
| 0						| Keep right     								|
The second image
|:---------------------:|:---------------------------------------------:| 
| 1    					| Turn right ahead  							| 
| 0    					| Keep left 									|
| 0						| No vehicles									|
| 0	      				| Ahead only					 				|
| 0						| Stop     										|
Rest of the images:
|:---------------------:|:---------------------------------------------:| 
| 0.9997    			| Children crossing  							| 
| 0.0002    			| Keep left 									|
| 0.0001				| End of all speed and passing limits			|
| 0	      				| Go straight or left			 				|
| 0						| End of speed limit (80km/h) 					|
|:---------------------:|:---------------------------------------------:| 
| 0.9998    			| Ahead only  									| 
| 0.0001    			| Turn left ahead 								|
| 0.0001				| Go straight or right							|
| 0.0001  				| Priority road			 						|
| 0						| Go straight or left 							|
|:---------------------:|:---------------------------------------------:| 
| 1    					| Keep right  									| 
| 0    					| Speed limit (20km/h) 							|
| 0						| Turn left ahead								|
| 0  					| Priority road			 						|
| 0						| Speed limit (120km/h) 						|
|:---------------------:|:---------------------------------------------:| 
| 1    					| Road work  									| 
| 0    					| Slippery road 								|
| 0						| End of speed limit (80km/h)					|
| 0  					| Keep left			 							|
| 0						| Beware of ice/snow 							|
|:---------------------:|:---------------------------------------------:| 
| 1    					| Yield  										| 
| 0    					| Priority road 								|
| 0						| Keep right									|
| 0  					| Go straight or right 							|
| 0						| No vehicles 									|
|:---------------------:|:---------------------------------------------:| 
| 1    					| No entry  									| 
| 0    					| Stop 											|
| 0						| Ahead only									|
| 0  					| Turn left ahead 								|
| 0						| Dangerous curve to the right 					|
|:---------------------:|:---------------------------------------------:| 
| 1    					| Ahead only  									| 
| 0    					| Go straight or left							|
| 0						| Turn right ahead								|
| 0  					| Roundabout mandatory 							|
| 0						| 'Go straight or right 						|


