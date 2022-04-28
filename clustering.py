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
def convert_into_tfidf_matrix(labels):
  #articles = df['column_name_of_article']
  #Create a TFIDF vectorizer to convert words to vectors
  vectorizer = TfidfVectorizer(max_features=500,
                            use_idf=True,
                            stop_words='english',
                            tokenizer=nltk.word_tokenize)
  #apply the vectorizer to the input texts
  #X = vectorizer.fit_transform(articles)
  X = vectorizer.fit_transform(labels)
  return X

#NOT really needed
'''
def print_vector_features(vectorizer):
    print(vectorizer.get_feature_names()
'''
#Task 1: Create a K means CLustering algorithm

def train_cluster(X): #X = tdidf_matrix
    km = KMeans(n_clusters=16, init='k-means++', max_iter=300,
           verbose=0, random_state=3425)
    
    km.fit(X)
    return km

# Task 2. a function to compute both intrinsic (Silhouette coefficient) and extrinsic
def compute_score(km):
  #km = train_cluster(X)        
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
  print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, km.labels_))

# When no ground truth is available
  print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))

# Task 3. a function to visualise those metrics values for values of N ranging from 2 to 16.    
def visualise_cluster(X, km):
    #X = tfidf_matrix, clusters = km.labels_ 
    dist = 1 - cosine_similarity(X)

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
    
    
    plt.show() #show the plot


if __name__ == "__main__":    
  df = shuffle(df)
  labels = df['text'] #'text' or name of column to cluster
  #Task 1 - Train
  X = convert_into_tfidf_matrix(labels)

  #Train Kmeans cluster
  km = train_cluster(X)
  #km.fit(X)

  #Task 2 - Scores
  compute_score(km)

  #Task 3 - Visualise
  visualise_cluster(X, km)
  #X = tfidf_matrix, clusters = km.labels_
