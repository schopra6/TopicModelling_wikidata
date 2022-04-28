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
