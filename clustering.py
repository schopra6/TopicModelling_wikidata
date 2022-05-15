import argparse
import os
from collections import Counter
from itertools import product
from statistics import median, mean

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from sklearn.cluster import KMeans
from sklearn import metrics
#visulaisation

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
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



    def convert_into_tfidf_matrix(texts):
      vectorizer = TfidfVectorizer(max_features=500,
                                use_idf=True,
                                stop_words='english',
                                tokenizer=nltk.word_tokenize)
    
      X = vectorizer.fit_transform(texts)
      return X

 
    def evaluate(self, labels):
   
        if not self.model:
            raise NotFittedError('Method train should be called first')
        predicted = self.model.labels_
        homogeneity = metrics.homogeneity_score(labels, predicted)
        completeness = metrics.completeness_score(labels, predicted)
        v_measure = metrics.v_measure_score(labels, predicted)
        randscore = metrics.adjusted_rand_score(labels, predicted)
        silhouette = metrics.silhouette_score(self.X, predicted)

        return {'Homogeneity': homogeneity, 'Completeness': completeness, 'V measure': v_measure,
                'Adjusted Rand Index': randscore, 'Silhouette coefficient': silhouette}

    def visualise_cluster(self, X, km):
        #X = tfidf_matrix, clusters = km.labels_ 
        dist = 1 - cosine_similarity(self.X)

    # Use multidimensional scaling to convert the dist matrix into a 2-dimensional array 
        MDS()
        # n_components=2 to plot results in a two-dimensional plane
        # "precomputed" because the  distance matrix dist is already computed
        # `random_state` set to 1 so that the plot is reproducible.
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]

        #set up colors per clusters using a dict
        # #1b9e77 (green) #d95f02 (orange) #7570b3 (purple) #e7298a (pink)
        #Label categories
        cluster_colors = {0: '#f0140c', 1: '#ad7144', 2: '#f5b92f', 3: '#e8f007', 4: '#88e014', \
                      5:"#0eedb2", 6:"#0dafdb", \
                      7:"#1330ed", 8:"#9a09e8", 9:"#e605b1", 10:"#c4a29d", 11:"#695232", 12:"#f7f088", 13:"#7e8778", \
                      14:"#7dada2", 15:"#628cf5"}

        #set up cluster names using a dict
        #cluster_names = {0: 'techno'}

        #some ipython magic to show the matplotlib plots inline
        %matplotlib inline 

        #create data frame that has the result of the MDS plus the cluster numbers and titles
        #km.labels_ == clusters
        df = pd.DataFrame(dict(x=xs, y=ys, label=km.labels_))

        #group by cluster
        groups = df.groupby('label')

        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                #    label=cluster_names[name], 
                    color=cluster_colors[name], 
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
            ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelleft=False)

        ax.legend(numpoints=1)  #show legend with only 1 point
        plt.xlabel('number of clusters and method')
        plt.ylabel('quality')
        plt.title('Metrics')
        plt.legend()

        plt.show() #show the plot
        plt.savefig('data/Clustering visualization.png')
        
  

    def visualise(results: dict):
        """
        Function for the results visualization. Stores the plot with the results into the folder named 'data'
        :param results: dictionary of dictionaries with 5 scores for each setting
        """
        res_df = pd.DataFrame(results)
        res_df.to_csv('C:/Users/Colmr/Desktop/dsprojject/Clustering results.csv', index=False)
        for metric in list(results.values())[0]:
            x = []
            y = []
            for result in results:
                x.append(result)
                y.append(results[result][metric])
            plt.plot(x, y, label=metric)
        plt.gcf().set_size_inches(10, 5)
        plt.xlabel('number of clusters and method')
        plt.ylabel('quality')
        plt.title('Metrics')
        plt.legend()
        plt.savefig('C:/Users/Colmr/Desktop/dsprojject/Clustering visualization.png')


def main(inputpath):
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
            results = clusterizer.evaluate(df['Person'])
        all_results[f'{option[0]}clust.'] = results
    Clusterizer.visualise(all_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clusterizer")
    parser.add_argument("--input", type=str, default='data/preprocessed_data.csv',
                        help="path to preprocessed data csv file")
    args = parser.parse_args()
    main(args.input)

    
