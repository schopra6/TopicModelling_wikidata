import pandas as pd
import numpy as np
from sklearn.datasets import load_files
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df1 = df1.sample(frac=1)
#X = the features used for classification
#y is the target variables
X = df1['Text']
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
y = df1.Target
#splitting the data to training and test data in the ratio 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Vectorize the input (X)
tfidf_vectorizer = TfidfVectorizer(max_features=400,
                                   use_idf=True,
                                   stop_words='english',
                                   tokenizer=nltk.word_tokenize,
                                   ngram_range=(1, 3))

X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_vec = tfidf_vectorizer.transform(X_test)

from sklearn.linear_model import Perceptron
perceptron_clf = Perceptron()
perceptron_clf.fit(X_train_vec, y_train)

y_pred = perceptron_clf.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Tuning using grid search cross-validation
# Create an object GridSearchCV

parameters = [a for a in np.linspace(0.01,1,11)]
clf = GridSearchCV( estimator=MultinomialNB(), 
                   param_grid={'alpha':parameters},
                   scoring='accuracy',
                   return_train_score=True,
                   cv=5
                  )
clf.fit( X_train_tf, Y_train )
print("Best score: %0.3f" % clf.best_score_)


# Predict the labels of the test instances
y_pred = clf.predict( X_test )


# Print the classification report
print(classification_report(y_test, y_pred ))
# Print the confusion matrix
print( confusion_matrix(y_test, y_pred ) )