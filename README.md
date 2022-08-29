#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./testimage/images1.jpeg "Traffic Sign 1"
[image5]: ./testimage/images2.jpeg "Traffic Sign 2"
[image6]: ./testimage/images3.jpeg "Traffic Sign 3"
[image7]: ./testimage/images4.jpeg "Traffic Sign 4"
[image8]: ./testimage/images5.jpeg "Traffic Sign 5"


[MD_image1]: ./MD_image/Speedlimit.png "Traffic Sign 5"

---


### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?

    27839

* The size of the validation set is ?

    6960

* The size of test set is ?

    12630

* The shape of a traffic sign image is ?

    (32, 32)

* The number of unique classes/labels in the data set is ?

    43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

    Speed limit (30km/h)
![alt text][MD_image1]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

Shuffled data set before training my NN. For those images downloaded from web, I have resized them into 32x32x3 before feed them into the model.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used architecture similar to LeNet  with dropout.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   						
| Convolution 3x3     	| 5x5 filter, 1x1 stride, outputs 28x28x6 	
| RELU					|										
| Max pooling	      	| 2x2 stride,  outputs 14x14x6
| Convolution 3x3	    | 5x5filter, 1x1 strides, outputs 10x10x16
| RELU					|	
| Max pooling	      	| 2x2 filter, 2x2 strides, Outputs 5x5x6
| Flatten               | 5x5x6 input, output 400
| Fully connected		| 400 input, output 120.  
| DropOut               | keep_rate 0.75  
| Fully connected    	| 120 input output 84
| DropOut               | keep_rate 0.75  
| Fully connected       | 84 input, output 43
										
 


#### 3. How did I train the model:

I used following setting to train the model.

epochs size:100  
batch size: 128  
learning rate:0.0001  
Keep rate: 0.75  
Optimizer: Adam Optimizier  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. I

I had some overfitting issue at the very beginning. During training the accuray increase first then drop to below 0.93. Then I add drop out to the network. The final accuray increases to 0.956 for validation set.

My final model results are:
* training set accuracy of ?
    0.987
* validation set accuracy of ? 
    0.964
* test set accuracy of ?
    0.888


If a well known architecture was chosen:
* What architecture was chosen?
I used LeNet architecture.
* Why did you believe it would be relevant to the traffic sign application?
Because I know it works well with digit images recognition. Traffic sign classification is similar to this problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
 My final model results are:  
    training set accuracy of 0.966  
    validation set accuracy of 0.948  
    test set accuracy of 0.872  
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]  
![alt text][image5]   
![alt text][image6]  
![alt text][image7]  
![alt text][image8]

The images have higher resolution than 32x32, they are resized to 32x32 before feeding into the model. The third image might be difficult to classify because its background may confuse the trained neuron network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory      		| Roundabout mandatory  									| 
| Children crossing     			| Children crossing 										|
| Keep Right					    |  Keep Right			
|Bumpy Road			                | Bumpy Road|
| Yield			                    | Yield    	
|


The model was able to correctly guess 4 of the 5 traffic signs when I set epoch = 30 and learning rate = 0.001, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.868.

After I increase epoch = 100 and reduce learning rate = 0.0001. The model can predicted all the 5 download traffic signs correctly, giving an accuracy of 100%, while the the accuracy on the test set of 0.888.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1), and the image does contain a Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Roundabout mandatory   									| 
| 3.25e-15     				| Speed limit (70km/h) 										|
| 2.72e-16					| Go straight or right											|
| 2.37e-17	      			| Go straight or left					 				|
| 1.87e-18				    | Speed limit (20km/h)  |  
|   		


For the first image, the model is relatively sure that this is a Children crossing (probability of 0.998), and the image does contain a Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.998         			| Children crossing   									| 
| 2.07e-3     				| Bicycles crossing 										|
| 9.61e-16					| Road narrows on the right											|
| 4.84e-6	      			| Double curve					 				|
| 1.22e-7				    | Traffic signals |  
|   		


