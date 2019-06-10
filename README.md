# Overview
This project uses an advanced image stitching algorithm (implementation details uploaded in PDF). The program also includes searching capabilities for finding same scene images and automatic stitching. Just specify a folder for the program to explore and a result will be generated if matches are found. 

Jobs are submitted via SQS - Simple Queue Service. Messages on the queue are automatically ingested via a Lambda Function (using a Lambda Trigger). The job requires the S3 bucket name, source folder, and destination folder in order to do a search and merge task. A CloudFormation template has been provided to help automate the deployment process.
 
 See the uploaded PDF for implementation details and experimental results. All images used in the paper can be found in the **src/examples** directory.
 
 <p align="center">
  <img width="300" src="/src/examples/tree-mrc/image1_image2_image3.jpg"><br><b>Example of 'tree-mrc'</b>
</p>
 
 ## Contributors
<a href="https://github.com/AndrewAltimit/Image-Stitcher/graphs/contributors">
  <img src="https://contributors-img.firebaseapp.com/image?repo=AndrewAltimit/Image-Stitcher" />
</a>

## Required Libs and Environment
* Python 3.6
* OpenCV 3.4.2 for Swift Features
* Numpy
* matplotlib (optional)


## AWS Services
* SQS
* Lambda
* S3
* CloudFormation
* CloudWatch
* IAM



## Serverless AWS Deployment

###### Inside the "AWS Deployment" folder you'll find the CloudFormation template, Lambda Function source code, custom Lambda Layers (OpenCV, NumPy), and the required S3 bucket contents.

1. Upload the contents of **Bucket Contents** to your S3 bucket.

2. Create a new stack in CloudFormation and upload **template.json**.

3. Specify a new IAM username, new queue name, your existing bucket name, and Lambda Layer ARNs for OpenCV and NumPy.
 
    OpenCV / NumPy are custom layers I compiled myself. I provided the zips in the "Lambda Layers" directory so you can upload them and use them as needed.    
    
4. Once the deployment is finished you can try it out by submitting a new message on the queue


****SQS Message Body****

     	 {
			"sourceFolder" : String,
			"bucketName" : String,
			"destFolder" : String
		}



## Running Locally

****Train Network and Classify Test Samples****

The src directory contains main.py which is used for image stitching.

	python main.py <image directory>
	
The program will search for potential matches and display the results. Matplotlib is used to show the Sift keypoints and how they are mapped between images. 
