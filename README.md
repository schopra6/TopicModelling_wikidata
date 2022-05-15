## UE 803: Data Science
# Project: Clustering and Classifying Articles based on Text and knowledge-base information

## About this project
In this project, we need to collect information about articles belonging to different categories (such as airports, artists, politicians, sportspeople, etc). Based on this information, we will try to automatically cluster and classify these articles into the correct categories. The source of information we are using are:
* Wikipedia Online Encyclopedia
* Wikidata knowledge base

## Installation
Note: Following code has been implemented in Python3

1. cd to the directory where ```requirements.txt``` is located;

2. run: `pip install -r requirements.txt` .

## How to execute the code
In order to execute the code from this repository and get the results, the following commands need to be run:

`python data_extraction.py num_articles 100 num_sentences 5`

`python preprocessing_nltk.py --input /extracted_data.csv/ --output data/preprocessed_data.csv/`

`python clustering.py --input data/preprocessed_data.csv --output data/`
  
`python category_classification.py --inputpath data/preprocessed_data.csv`


## General Overview
The project constitues of four steps, namely:
1. Corpus Extraction
2. Pre-Processing
3. Clustering
4. Classifying


### Corpus Extraction
This step extracts information about articles from wikipedia and wikidata. These articles belong to the categories:
* Airports
* Artists
* Astronauts
* Building
* Astronomical_objects
* City
* Comics_characters
* Companies
* Foods
* Transport
* Monuments_and_memorials
* Politicians
* Sports_teams
* Sportspeople
* Universities_and_colleges
* Written_communication'.

The script used by the Corpus Extraction gets k (**```num_articles```** param) articles from each category and stores the following features such as: description, page content (where the number n of sentences is represented by the **```num_sentences```** param) , infobox, wikidata statements (triples). The result is stored in the **```data/extracted_data.csv```**.

### Pre-Processing
This step takes, as an input, the csv file (**```data/extracted_data.csv```**) generated by the previous step. The data is processed using the following steps:
* Tokenize the text
* Lowercase the tokens
* Remove punctuation and function words
* Remove stop words

The preprocessed output is stored in **```data/preprocessed_data.csv```**

### Clustering
This step takes the **```data/preprocessed_data.csv```** as input which is going to be used to train the KMeans algorithm (**```--input```** param). The method used in order to process the data is:
* Tf-idk

The results obtained are stored in **```data/Clustering results.csv```** (which shows a comparison table) and **```data/Clustering visualization.png```** (which shows a comparison plot).

### Classifying
This step takes the **```data/preprocessed_data.csv```** as input which is going to be used to train the Perceptron algorithm. The method used to preprocess the data is Tf-Idf Vectorizer. The results obtained are stored in the ```data``` directory. The files obtatined are:

* `Confusion matrix 16 classes.png` with the confusion matrix for the 16 class classification.
* `Scores 16 classes.csv` with the Precision, Recall, F1-Score and Support scores for the 16 class classification

## Results

The results are obtained on the data stored in the ```data``` directory:

**Clustering**:

![Clustering visualization](https://github.com/schopra6/TopicModelling_wikidata/blob/main/data/Clustering%20visualization.png)

|16clust.            |15clust.            |14clust.            |13clust.            |12clust.            |11clust.            |10clust.            |9clust.             |8clust.             |7clust.             |6clust.             |5clust.             |4clust.             |3clust.             |2clust.             |
|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
|0.6680916223497617  |0.6983192400534757  |0.6696884923205174  |0.5906099312548578  |0.6203303978577499  |0.5870465175217298  |0.5574005499406839  |0.5437336837606537  |0.4829662450603848  |0.4572132727484873  |0.38920056491263944 |0.32889613864546835 |0.26324447104990567 |0.17971181188393984 |0.10768322691491419 |
|0.7058643137837441  |0.725532305195805   |0.754295664076981   |0.7083097655869558  |0.7348225249553275  |0.7513919747278484  |0.7560636464252114  |0.734479465375821   |0.7404350905295765  |0.7572583291106593  |0.7177446756717246  |0.6972722112277343  |0.7069819947325824  |0.6561038446101938  |0.6751243629217271  |
|0.686458746095857   |0.7116657206139825  |0.7094785763874869  |0.644127243551073   |0.6727399418718589  |0.659129343054648   |0.6417080777284814  |0.6248742248116411  |0.5846080840046943  |0.5701715189366625  |0.5047162641704035  |0.4469639662652354  |0.3836405371504451  |0.28214262267709606 |0.18574058532942533 |
|0.42997955192211296 |0.5637443909527183  |0.4389854197472739  |0.3497149795036718  |0.4168178664403366  |0.34107045621984067 |0.3108731459727958  |0.3337065053084658  |0.23979809632333296 |0.2193361190743175  |0.17080162974321872 |0.1311890072602876  |0.08791649467791553 |0.052895912354631265|0.03052442766771767 |
|0.09795952708393732 |0.09715241698108781 |0.09428528677153968 |0.09372228136789948 |0.0906886180163463  |0.08619948105220934 |0.08340299746693836 |0.08255049231214943 |0.08114423009222613 |0.0801217518314597  |0.0731632680503235  |0.06620950570426505 |0.05949654226982902 |0.0517310929017951  |0.0392090136351276  |


**Classification**:

**16 class classification:**

|**category**                  |**precision**|**recall**|**f1-score**|**support**|
|------------------------------|---------|------|--------|-------|
|Airports                      |1.0      |0.75  |0.86    |28.0   |
|Artists                       |0.86     |0.86  |0.86    |14.0   |
|Astronauts                    |0.77     |1.0   |0.87    |17.0   |
|Astronomical_objects          |1.0      |1.0   |1.0     |9.0    |
|Building                      |0.73     |0.73  |0.73    |11.0   |
|City                          |1.0      |0.95  |0.97    |20.0   |
|Comics_characters             |0.81     |0.93  |0.87    |14.0   |
|Companies                     |0.83     |0.71  |0.77    |7.0    |
|Foods                         |0.87     |1.0   |0.93    |13.0   |
|Monuments_and_memorials       |1.0      |0.89  |0.94    |9.0    |
|Politicians                   |0.86     |0.67  |0.75    |9.0    |
|Sports_teams                  |0.71     |0.59  |0.64    |29.0   |
|Sportspeople                  |0.38     |0.44  |0.41    |18.0   |
|Transport                     |0.75     |0.75  |0.75    |8.0    |
|Universities_and_colleges     |0.55     |0.75  |0.63    |8.0    |
|Written_communication         |0.67     |1.0   |0.8     |4.0    |
|accuracy                      |0.79     |0.79  |0.79    |0.79   |
|macro avg                     |0.8      |0.81  |0.8     |218.0  |
|weighted avg                  |0.81     |0.79  |0.79    |218.0  |

## Bonus Analysis
![Confusion matrix 16 classes](https://github.com/schopra6/TopicModelling_wikidata/blob/main/data/Confusion%20matrix%2016%20classes.png)

We did a visualisation on the confusion matrix to identify which category was misidentified and to what. This shows us the reason for low overall scores for sportspeople. 
Out of 17 test examples for Sportspeople, 9 were identified as Sports_teams category. This is because the word correlation of text from sportspeople with Sports_teams was also probable. Since number of training samples were more in sports_team, so it labelled it sports_team.




##Created By
* Sahil Chopra
* Jorge Vasquez
* Colm Rooney