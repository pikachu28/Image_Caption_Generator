# Image_Caption_Generator


To make our image caption generator model, we will be merging CNN-RNN model architecture.
-CNN is used for extracting features from the image. We will use the
pre-trained model Xception.
-LSTM will use the information CNN to generate a description of the
image

## Data Description:

### Data Collected From:

https://www.kaggle.com/wikiabhi/image-caption-generator?



We have an image folder in which all the images are stored with their name as their unique ID.
And we have an image description folder(Description_of_image) in which several files. Train , test, dev files are there to divide images into training and testing. They are containing image ID from images folder. Description of Images is in Tokenized and Text Normalized file. Each image id or image have 5 descriptions in those files.
We have 6000 images for training.


### Using a pretrained library: Xception

We are going to use a pretrained library for training our model. It will ease our work as we do not have teach our model everything from beginning. Basics are done using Xception and then training our model on our dataset and after this we follow our cycle of hyperparameter tunning. Everytime we tune a hyperparameter we do not have to start from beginning, we can just start from where we train our model this technique save alot of time and efforts.

#### But what is Xception Model and Transfer Learning?

Xception is a convolutional neural network that is 71 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 299-by-299.

You can use classify to classify new images using the Xception model.
This technique is called Tranfer Learning.This is commonanly used in Deep Learning Applications.
You can take a pretrained network and use it as a starting point to learn a new task. Fine-tuning a network with transfer learning is usually much faster and easier than training a network from scratch with randomly initialized weights. You can quickly transfer learned features to a new task using a smaller number of training images.


<img width="1128" alt="Screenshot 2020-09-30 at 3 37 14 AM" src="https://user-images.githubusercontent.com/62153950/94621641-4e977780-02ce-11eb-918d-57a7d3e98d3b.png">

#### Learn More about Xception from:

https://in.mathworks.com/help/deeplearning/ref/xception.html

https://medium.com/analytics-vidhya/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206


### Data Cleaning

The format of our file is image and caption seperated by a new line('\n')
We are going to store our final cleaned data in descriptions.txt

For loading ,Cleaning and saving  our data we will use five functions:

              1. load_doc(filename) - For loading the document file and reading the contents inside the file into a string (loading tokenized file)
              
              2. all_image_caption(filename) - This function will create a descriptions dictionary that maps images with a list of 5 captions.
              
              3. cleaning_text(descriptions) - This functions takes all descriptions and performs data cleaning. This is an important step when we work with textual data, according to our goal, we decide what type of cleaning we want to perform on the text. In our case, we will be                                                removing punctuations, converting all text to lowercase and removing words that contain numbers.

              4. text_vocabulary(descriptions) - This is a simple function that will separate all the unique words and create the vocabulary from all the descriptions.
              
              5. save_descriptions(descriptions, filename) - This function will create a list of all the descriptions that have been preprocessed and store them into a file (descriptions.txt). 


### Feature Extraction

This is the time to use Transfer Learning. We don't have to do everything on our own we will use pre-trained model Xception to train it on our data.
One thing to remeber is that Xception Model take 299*299*3 as image input size. We will remove the last classification layer and get 2048 feature vectors.


#### extract_features():

The function extract_features() will extract features for all images and we will map image names with their respective feature array. Then we will dump the features dictionary into a “features.p” pickle file.

### Loading Dataset For Training the model:

We have Flickr_8k.trainImages.txt file that contains a list of 6000 image names that we will use for training.

Functions to load Dataset:

    1. load_photos(filename): This will load in a string and will return list of the image names.
    2. load_clean_description(filename, photos): This function will create a dictionary that contains captions for each photo from the list returned from above function.We also append the <start> and <end> identifier for each caption. We need this so that our LSTM model can identify the starting and ending of the caption.
    3. load_features(photos): Give us the dictionary for image name and their feature vector which we have previously extracted from Xception Model.

