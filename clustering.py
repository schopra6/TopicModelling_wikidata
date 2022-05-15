# import modules
import argparse
import os
from collections import Counter
from itertools import product
from statistics import median, mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import word_tokenize
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

'''
1. a function to train a clustering algorithm on some data using N
clusters

2. a function to compute both intrinsic (Silhouette coefficient) and
extrinsic (homogeneity, completeness, v-measure, adjusted Rand
index) evaluation scores for clustering results.

3. a function to visualise those metrics values for values of N ranging
from 2 to 16.

'''
class Clusterizer:
    
    '''
    Class to perform clustering on the preprocessed dataFrame
    '''
    def __init__(self):
        self.model = None
        self.X = None
        self.method = None


    def train(self, texts, descriptions, n_clusters, method):
        rawdata_X = [f'{descr} {text}' for text, descr in zip(texts, descriptions)]
        self.method = method
        self.X = self.convert_into_tfidf_matrix(rawdata_X)
        print(f'▶ Model being trained with {n_clusters} clusters using vectorizer-matrix')

        # initializing the model, with n clusters and fitting the tfidf matrix
        print(">>>>> MODEL TRAINING COMPLETE ...")
        self.model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300,
               verbose=0, random_state=3425)
        self.model.fit(self.X)



    def convert_into_tfidf_matrix(self, texts):
      vectorizer = TfidfVectorizer(max_features=500,
                                use_idf=True,
                                stop_words='english',
                                tokenizer=nltk.word_tokenize)
    
      X = vectorizer.fit_transform(texts)
      return X

 
    def compute_scores(self, labels):
   
        if not self.model:
            raise NotFittedError('Train model before evaluaing scores.')
        predicted = self.model.labels_
        homogeneity = metrics.homogeneity_score(labels, predicted)
        completeness = metrics.completeness_score(labels, predicted)
        v_measure = metrics.v_measure_score(labels, predicted)
        randscore = metrics.adjusted_rand_score(labels, predicted)
        silhouette = metrics.silhouette_score(self.X, predicted)

        return {'Homogeneity': homogeneity, 'Completeness': completeness, 'V measure': v_measure,
                'Adjusted Rand Index': randscore, 'Silhouette coefficient': silhouette}
  

    def visualise_cluster(results, outputpath):
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{outputpath}/Clustering results.csv', index=False)
        for metric in list(results.values())[0]:
            x = []
            y = []
            for result in results:
                x.append(result)
                y.append(results[result][metric])
            plt.plot(x, y, label=metric)
        plt.gcf().set_size_inches(10, 5)
        plt.xlabel('number of clusters')
        plt.ylabel('quality')
        plt.title('Metrics')
        plt.legend()
        plt.savefig(f'{outputpath}/Clustering visualization.png')


def main(inputpath, outputpath):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
        df = shuffle(df).reset_index(drop=True)
    else:
        raise FileNotFoundError(f'Preprocessed data at {inputpath} not found. Please ensure your pathname is correct.')
    
    all_results = dict()
    for option in product([16, 15,14,13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]):
        clusterizer = Clusterizer()
        clusterizer.train(df['Preprocessed Wikipage'], df['Preprocessed Description'], option[0], 'vectorizer_matrix')
        if option[0]:
            results = clusterizer.compute_scores(df['Person'])
        all_results[f'{option[0]}clust.'] = results
    Clusterizer.visualise_cluster(all_results, outputpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clusterizer")
    parser.add_argument("--input", type=str,
                        help="Please add path to preprocessed data csv file")
    parser.add_argument("--output", type=str,
                        help="Please add path to save clustering results")
    args = parser.parse_args()
    main(args.input, args.output)

    
    
