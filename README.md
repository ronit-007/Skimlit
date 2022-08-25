# Skimlit 📄🔥



## Table of Content
  * [Demo](#demo)
  * [Motivation](#motivation)
  * [SkimLit ](#skimlit)
  * [Getting Data ready](#getting-data-ready)
  * [Creating and training series of models](#creating-and-training-series-of-models)
  * [Compare model results](#compare-model-results)
  * [Evaluating the model](#evaluating-the-model)
  * [Making example perdictions on custom examples](#making-example-perdictions-on-custom-examples)
  * [Link to model](#link-to-model)
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

- To make sure the readability of the dataset I preferred to make the data look like this
  
    ```
     [{'line_number': 0,
     'target':'Background',
     'text':'Emotional eating is associated with overeating and the development of obesity .\n',
     'total_lines':11}...]
    ```
 - Making numeric labels (ML Model requires labels)
 - Label encoding labels
 

### Preparing our data (the text) for deep sequence models

For deep sequence models we need to create vectorization and embedding layers
#### Converting text into numbers

When dealing with the text problem, one of the first thingd we'll have to do before we can build a model is to convert the text to numbers.

There are few ways to do this, namely :
* **Tokenization** - direct mapping of tken (a token could be a word or a character) to number
* **Embedding** - create a matrix of feature vector for each token (the size of the feature vector can be defined and this embedding can be learned)

## Creating and training series of models 
### Model 0(Baseline)

As with all machine learning modelling experiments , it's important to create a baseline model so we've got a benchmark for the feature experiments to build upon.

To create the baseine , we'll use Sklearn's Multinomial Naive Bayes using the TF-IDF formula to convert our words to numbers.

>**Note:** It's common to use non-DL algorithms as a baseline because of their spped and then later using DL to see if we can improve upon them

### Model 1(Conv1D with token embedding)

A Conv1D network model built with keras functional api.

### Model 2(Feature extraction with pretrained token embedding)
Used pretrained word embedding from TensorFlow Hub, more specifically the universal sentence encoder(USE):
https://tfhub.dev/google/universal-sentence-encoder/4

### Model 3(Conv1D with character embedding)

The paper which we replicated stated that they used combination of token and character embeddings.

Previously we've token-level embedding but we'll need to do similiar steps for characters if we want to use char-level embeddings

### Model 4(Combining pretrained token embedding + characters embedding (hybrid embedding layers))

The buliding of model involved:
  1. Create a token-level embedding (similar to model_1)
  2. Create a character-level model (similar to model_3 with a slight modification)
  3. Combine 1 & 2 with  a concatenate (`layers.Concatenate`)
  4. Build a series of output layers on the top.
  5. Construct a model which takes token and character-level sequence as input and produces sequence label probabilities as output
 
**Plot of the model**:
   <img src="https://user-images.githubusercontent.com/106836228/186617358-8beb2f86-d2d1-461d-8cba-88516bb6d470.png">
   
### Model 5(Transfer learning with pretrained token embeddings + character embeddings + positional embeddings)
 
   > **Note:** Any engineered features used to train a model need to be availabel at the testing time. In our case line numbers and total lines are availabel
 
 The Building of the model involved:
   1. Create a token-level Model
   2. Create a character-level Model
   3. Create a model for the "line_number" feature
   4. Creae a model for the "total_lines" feature
   5. Combine the outputs of 1 & 2 using `tf.keras.layers.Concatenate`
   6. Combine the outputs of 3,4,5 using `tf.keras.layers.Concatenate`
   7. Create an output layer to accept the tribrid embedding and output label probabilties
   8. Combine the inputus of 1 ,2 ,3,4 and outputs of into a `tf.keras.Model`
    
   **Plot of the model**:
    <img src="https://user-images.githubusercontent.com/106836228/186620260-76cb1a7c-db0b-4873-88c4-00ef3ee83abc.png">
    
## Compare model results

After going through experimentation the best performing model is choosen.

<img src="https://user-images.githubusercontent.com/106836228/186621101-175cec89-f57e-41ca-80e1-26574afb28f0.png">

Using the f1 score only:

<img src="https://user-images.githubusercontent.com/106836228/186621115-258628dd-3987-4729-b485-2ff5c6069239.png">


## Evaluating the model

The model evaluation was done on the test dataset after the best performing model is chosen and saved.
The model was reloaded and evaluated after creating the test dataset and prefetching to optimize the GPU

After the evalaution we found where the model predicted the wrong ones and tried to find the reason. Whether the model predicted wrong because of overfitting , or our model couldn't understand the pattern or the label was incorectly marked etc.

## Making example perdictions on custom examples

Since the abstract we are going to use aren't the same format that the model is trained on, so we need to preprocess the data.

Therefore for each abstract we need to :
1. Split it into sentences (lines).
2. Split it into characters.
3. Find the number of each line.
4. Find the total number of lines.

Then we can make prediction.





## Link to model

The model is in gdrive:- https://bit.ly/3Kjl547
 


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)
 [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730141-b8e739bb-8c0e-42ce-bc83-81f45cde875b.png" width=200>](https://www.tensorflow.org/)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730270-20281dad-529e-46b9-8a2d-385a6b46b32f.png" width=200>](https://www.tensorflow.org/api_docs/python/tf/keras)
 
 
 
 ## Future Scope

* Use multiple Algorithms
* Deploying the model
* Front-End
