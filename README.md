# TopicModelling_wikidata
## clustering and Classifying Articles based on Text and knowledge-base information


Clustering (8 points)
The goal of this exercise is to use the collected data (text and wikidata descriptions) to automatically cluster the Wikipedia documents
first, using 16 clusters and second, experimenting with different numbers of clusters.
Your code should include the following functions:
• a function to train a clustering algorithm on some data using N
clusters
• a function to compute both intrinsic (Silhouette coefficient) and
extrinsic (homogeneity, completeness, v-measure, adjusted Rand
index) evaluation scores for clustering results.
• a function to visualise those metrics values for values of N ranging
from 2 to 16.


## Installation
Note: Following code has been implemented in Python3

1. cd to the directory where ```requirements.txt``` is located;

2. run: `pip install -r requirements.txt` .

# Classification

**```classification.py```** script  takes file path as input , preprocesses using Tf-Idf Vectorizer and train a single layer perceptron model to predict classes. The evaluation results are as follows 

- ```Confusion matrix 16 classes.png``` with the confusion matrix for the 16 class classification


- ```Scores 16 classes.csv``` with the Recall, Precision and F-score computed for the 16 class classification


## Results

The results are obtained on the data stored in the ```data``` directory:

**Classification**:

**16 class classification:**



