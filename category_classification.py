import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split

import nltk

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

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

class PerceptronClassifier:

    def __init__(self):
        self.model = None

    def split_vectorise_data(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15,
                                                            stratify=y)
        tfidf_vectorizer = TfidfVectorizer(max_features=400,
                                           use_idf=True,
                                           stop_words='english',
                                           tokenizer=nltk.word_tokenize,
                                           ngram_range=(1, 3))
        X_train_vec = tfidf_vectorizer.fit_transform(X_train)
        X_test_vec = tfidf_vectorizer.transform(X_test)

        return X_train_vec, X_test_vec, y_train, y_test

    def train(self, X, y):
        """
        Function used for training the Perceptron algorithm on the data vectorized using Tf-Idf vectorization
        :param X: sparse matrix with the texts preprocessed by Tf-Idf Vectorizer
        :param y: expected values for each sample
        """
        self.model = Perceptron(penalty='l1', alpha=0.001, random_state=0)
        self.model.fit(X, y)

    def predict(self, X):
        """
        :param X: sparse matrix with the texts preprocessed by Tf-Idf Vectorizer
        :return: predicted values
        """
        pred = self.model.predict(X)
        return pred
    def compute_scores(self, expected: list, predicted: list, num_classes: int):
        """
        Function used for computing the confusion matrix, Recall, Precision and F1 scores
        :param expected: list of expected values
        :param predicted: list of predicted values
        :param num_classes: number of classes
        :return: numpy array with the confusion matrix, string with the recall, precision and f1 scores
        """

        classes = self.model.classes_

        # get confusion matrix
        conf_matrix = confusion_matrix(expected, predicted)
        plt.figure(figsize=(10, 10))
        plot = sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.savefig(f'Confusion matrix {num_classes} classes.png')
        plt.clf()

        # get precision, recall, F1
        report = classification_report(predicted, expected)
        report_dict = classification_report(predicted, expected, output_dict=True)
        df = pd.DataFrame(report_dict).transpose().round(2)
        df.to_csv(f'Scores {num_classes} classes.csv')

        return conf_matrix, report, classes


def main(path: str):
    """
    :param path: path to the csv file
     """
    df = pd.read_csv(path)
    df = shuffle(df).reset_index(drop=True)
    classifier = PerceptronClassifier()
    X_train, X_test, y_train, y_test = classifier.split_vectorise_data(df['text'],df['category'])
    classifier.train(X_train, y_train)
    predicted = classifier.predict(X_test)
    classes = np.unique(y_train)
    conf_matrix, class_report, classes = classifier.compute_scores(y_test, predicted, classes)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument("--inputpath", type=str,
                        help="path to the csv file in required format")

    args = parser.parse_args()
    main(args.inputpath)
