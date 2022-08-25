# Skimlit 📄🔥



## Table of Content
  * [Demo](#demo)
  * [Motivation](#motivation)
  * [Classify Different Dog Breeds](#classify-different-dog-breeds)
  * [Getting Data ready](#getting-data-ready)
  * [Creating and training the Neural Network](#creating-and-training-the-neural-network)
  * [Performances and results](#performances-and-results)
  * [Improve the model](#improve-the-model)
  * [Architecture](#architecture)
  * [Technologies Used](#technologies-used)
  * [Future scope of project](#future-scope)


## Demo


The purpose  is to build an NLP model to make reading medical abstract easier.





## Motivation
What to do when you are at collage and having a strong base of mathematics and keen intrest in learning ML ,DL and AI? I started to learn Machine Learning model to get most out of it. I came to know mathematics behind models. Finally it is important to work on application (real world application) to actually make a difference.





## SkimLit 

The paper we're replicating (the source of the dataset we'll be using ) is available at: https://arxiv.org/abs/1710.06071



And reading through the paper above, we see that the model architecture that they use to achieve their best results is available here: https://arxiv.org/abs/1612.05251.

We're going to go through the following workflow:

#### 1. Problem

Automatically classifying each sentence in an abstract would help researchers read abstracts more efficiently, especially in fields where abstracts may be long, such    as the medical field.

#### 2.Data

Since we be replicated the paper above (PubMed 200k RCT),  downloading the dataset they used.

We can do so from the authors GitHub: https://github.com/Franck-Dernoncourt/pubmed-rct

#### 3. Evaluation

Evaluation to be taken to see how well our model performs so that we can be certain it functions approriatly.

#### 4. Features

Some information about the data:

- The dataset consists of approximately 200,000 abstracts of randomized controlled trials, totaling 2.3 million sentences. 
- Each sentence of each abstract is labeled with their role in the abstract using one of the following classes: background, objective, method, result, or conclusion. 


For preprocessing our data, we're going to use TensorFlow 2.x. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them.

## Getting Data ready
### Preprocessing data 

- To make sure the readability if the dataset I preferred to make the data look like this
  
    ```
     [{'line_number': 0,
     'target':'Background',
     'text':'Emotional eating is associated with overeating and the development of obesity .\n',
     'total_lines':11}...]
    ```
 - Making numeric labels (ML Model requires labels)
 - Label encoding labels

A good place to read about this type of function is the [TensorFlow documentation on loading images](https://www.tensorflow.org/tutorials/load_data/images). 

### Preparing our data (the text) for deep sequence models

For deep sequence models we need to create vectorization and embedding layers
#### Converting text into numbers

When dealing with the text problem, one of the first thingd we'll have to do before we can build a model is to convert the text to numbers.

There are few ways to do this, namely :
* **Tokenization** - direct mapping of tken (a token could be a word or a character) to number
* **Embedding** - create a matrix of feature vector for each token (the size of the feature vector can be defined and this embedding can be learned)

## Creating and training series of models 
### Model 0(Baseline)

Creating a model using sklearn TfidVectorizer , naive_bayes MultinomialNB and pipeline

### Model 1(Conv1D with token embedding)

A Conv1D network model built with keras functional api.

In this project, we're using the **`mobilenet_v2_130_224`** model from TensorFlow Hub.
https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
MobileNetV2 is a significant improvement over MobileNetV1 and pushes the state of the art for mobile visual recognition including classification, object detection and semantic segmentation. MobileNetV2 is released as part of TensorFlow-Slim Image Classification Library, or you can start exploring MobileNetV2 right away in Colaboratory. Alternately, you can download the notebook and explore it locally using Jupyter. MobileNetV2 is also available as modules on TF-Hub, and pretrained checkpoints can be found on github.
<img src="https://user-images.githubusercontent.com/106836228/186514899-12e8ca5a-0bb6-4c7d-a275-de8c064e1815.png">

### Setting up the model layers

The first layer we use is the model from TensorFlow Hub (`hub.KerasLayer(MODEL_URL)`. This **input layer** takes in our images and finds patterns in them based on the patterns [`mobilenet_v2_130_224`](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) has found.

The next layer (`tf.keras.layers.Dense()`) is the **output layer** of our model. It brings all of the information discovered in the input layer together and outputs it in the shape we're after, 120 (the number of unique labels we have). The `activation="softmax"` parameter tells the output layer, we'd like to assign a probability value to each of the 120 labels [somewhere between 0 & 1](https://en.wikipedia.org/wiki/Softmax_function). The higher the value, the more the model believes the input image should have that label. 


## Performances and results


## Predicted Label/Breed and probability distribution
Another way to analyse the inference of our model is to print out the predicted class of an image and it's probability distribution. For each image, we show it's original label (left), it's predicted label (right), and the probabilty assossiated with the predicted label (how much confident is our Neural Network about the predicted class).

<img src="https://user-images.githubusercontent.com/106836228/186514589-6897e186-8260-4db1-b685-147abd367a82.png">


# Improve the model
How to approuve model accuracy :
1. [Trying another model from TensorFlow Hub](https://tfhub.dev/) - A different model could perform better on our dataset. 
2. [Data augmentation](https://bair.berkeley.edu/blog/2019/06/07/data_aug/) - Take the training images and manipulate (crop, resize) or distort them (flip, rotate) to create even more training data for the model to learn from. 
3. [Fine-tuning](https://www.tensorflow.org/hub/tf2_saved_model#fine-tuning)

## Architecture

[![](https://imgur.com/fHwnL1y.png)]

[![](https://imgur.com/E1KYSAO.png)]
## Link to model:- [https://bit.ly/3Kjl547]

## Model:
https://imgur.com/auiBIYd
## Overview
This is a deep learning model which helps to read medical abstract easier.


## Motivation
What to do when you are at collage and having a strong base of mathematics and keen intrest in learning ML ,DL and AI? I started to learn Machine Learning model to get most out of it. I came to know mathematics behind all unsupervised models. Finally it is important to work on application (real world application) to actually make a difference.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)
 [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730141-b8e739bb-8c0e-42ce-bc83-81f45cde875b.png" width=200>](https://www.tensorflow.org/)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730270-20281dad-529e-46b9-8a2d-385a6b46b32f.png" width=200>](https://www.tensorflow.org/api_docs/python/tf/keras)
 
 
 
 ## Future Scope

* Use multiple Algorithms
* Deploying the model
* Front-End
