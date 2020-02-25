# README

## AIT 726 - Homework 1
### Authors: 
- Yasas
- Prashanti
- Ashwini

## Steps:
1. Place the dataset folder (`tweet`) containing test and train directories in the folder containing logistic_regression.py and naive_bayes.py.
2. You can execute each `.py` file by command `python <file_name>`.

# Answers to Bonus Questions

 1. how would the results change if you used term frequency instead of binary representation for both logistic regression and Naïve Bayes (1 point)?
 From the results populated from Naive Bayes we observe that term frequency has a slight better performance when compared to binary representation.
 where as its not making much difference in logistic regression. However we notice tf-idf is making a difference in logistic regression.
 
 For example during Naive Bayes execution we noticed 
 Running Stemming With Frequency BoW Features had an Accuracy of 89.72 , while Running Stemming With Binary BoW Features had an Accuracy of 89.22
 Running No Stemming With Frequency BoW Features had an Accuracy of 89.41, while Running No Stemming With Binary BoW Features had an Accuracy of 88.90
 Similarly during Logistic Regression execution we noticed
 Running Stemming With Frequency BoW Features had an Accuracy of 90.17, while Running Stemming With Binary BoW Features had an Accuracy of 90.44
 
 2. What about term frequency x inverse document frequency (1 point)?  
 From the results populated in logistic regression we can notice inverse document frequency implementation is slightly better than term frequency 
 
 For example during logistic regression we noticed. 
 Running Stemming With Frequency BoW Features had an Accuracy of 90.17 where as Running Stemming With TFIDF Features had an Accuracy of 90.53
 additional results can be found in logistic_regression_results.log
 
 3. How do your results change if you regularize your logistic regression (1 point)?
  we have implemented L2 regularization and noticed no much difference when we compared the accuracy results with and without regularization. 
  Results and implementation can be fond in logistic_regression.py and logistic_regression_results.log




