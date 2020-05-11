# README

## Aspect Based Sentiment Analysis

### AIT 726 - Project
   
### Authors: 
- Yasas
- Prashanti
- Ashwini

### Introduction

Abundance of consumer reviews on the internet helps most of the customers to decide the best product for their needs among the plethora of available brands and types of products or services. However, the amount of such reviews for a given product is numerous, and reading them manually by each person is a waste of time and money. Therefore, a need for an automated process for understanding these reviews to summarize the good and the bad of a given product or service is required. 
Sentiment Analysis alone addresses only part of the problem. A potential customer could be interested in only one or several aspects of the product or service. So we have built a model to perform aspect classification and polarity classification which will help identify the important aspect categories and provide respective polarity within the aspect classification. 

### Solution 
#### Part 1:
- We have created individual baseline models for each task ( Aspect category classification and polarity classification ).

1. Data loading -  For all the restaurant reviews we parsed the XML files and retrieved the reviews for SemEval’16 and foursquare datasets.

2. Preprocessing and Feature Extraction - Tokenization, and lemmatization of reviews is performed to get word tokens and their root words. Text vectorization is then performed using term frequency-inverse document frequency (TF-IDF) vectorizer with a ngram range of 1-3. In addition to TF-IDF features, we provided the Aspect Category as a feature when training the polarity classification model. Moreover, we provided predefined term categories (obtained from a vocabulary file) for each word in the input review sentence as features for aspect classification. 
 
3. Baseline Models: Following classifiers are implemented to predict aspects and corresponding sentiments in baseline models. 
    - Naive Bayes Classifier 
    - Logistic regression 

4. Cross-Validation and Error Analysis: Performed five-fold cross-validation on the training data and performed the error analysis using the predictions obtained in cross-validation. 

5. Train Models: Trained the models using the optimal hyper-parameters explored in the cross validation process on the whole training dataset. 

6. Evaluation: Evaluated the trained models on the test data (SemEval’16 and Foursquare datasets)

#### Part 2: Bi-LSTM neural network models are implemented for both aspect classification and polarity prediction.

1. Data loading:  For all the restaurant reviews we parsed the XML files and retrieved the reviews for SemEval’16 and foursquare datasets.

2. Preprocessing:  Train and test data are passed through the spacy_doc pipeline to generate the list of tokens, which are later passed through Keras tokenizer to convert them to integers. Finally, they are padded to max length. By performing the above steps we have our x_train, x_test, and x_test_fs input features. Created word embedding matrix and embedding_index for all the sentences based on word2vec.

3. i-LSTM model:

    - We have created Bi-LSTM models with one layer of 128 hidden units, and a fully connected output layer using sigmoid as an activation function. We have used Adam optimizer and cross-entropy loss function with a learning rate of 0.001 for all the models.
In addition to aspect classification, here in polarity classification, we are passing aspect dimension and aspect encoder to the concatenation layer.

4. Train with Validation: Split the Original Training data to Training and Validation splits and explore the optimal parameters by manually inspecting the performance of the models on validation data using different hyper-parameters.

5. Cross-Validation-Predict and Error Analysis: Performed five-fold cross-validation on the training data to predict the results of the Bi-LSTM model. Used the results to perform the error analysis against the results from the baseline approach. 

6. Train Models: Trained the models using the optimal hyper-parameters explored in the validation process on the whole training dataset. 

## Input and Output – Demo Screenshots using Bi-LSTM

- [Link](https://colab.research.google.com/drive/18jSbxsm6quaxHU-mTms5NroGDdJ0YHPj) for demo screenshots - 

In the demo, a summary function as a wrapper function is used to call all the pre-processing methods and pre-saved models to perform predictions on a review sentence as shown below:
 
- Aspect Classification
    - Input: restaurant review sentence.


- Polarity Classification
    - Input: restaurant review sentence and aspect.
