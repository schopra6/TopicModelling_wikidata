# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:57:09 2022

@author: Colmr
"""

import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
#nltk.download('stopwords')

'''
with open("/content/webnlg-test.txt") as infile:
  wikidata_txt = infile.read()
'''

#turning datafile string into sentence
def sentence_segmentation(wikidata_txt):
    sentences = nltk.sent_tokenize(wikidata_txt)
    [sentence for sentence in sentences]
    return sentences

#sentences = sentence_segmentation(wikidata_txt)

def flatten_list(tokens):
  flat_list = list()
  for sub_list in tokens:
    flat_list += sub_list
  print(flat_list)
  return flat_list

#tokenizing each sentence
def make_tokens(sentences):
  tokens = []
  for sentence in sentences:
    tokens.append(nltk.word_tokenize(sentence))
  #print(tokens)
  flatten_list(tokens)
  for token in tokens:
      if '.' in token:
          token + ' '
  return flatten_list(tokens)

#tokens = make_tokens(sentences)

#used to flatten the list of list of tokens into a list of tokens

  
#tokens = flatten_list(tokens)

#lower casing list

def make_lower_remove_stopwords_punc(tokens):
  #list comp for lowercasing
  tokens = [w.lower() for w in tokens]
  
  #helper function for removing punc in a single string
  def remove_punc(string):
    punctuations = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for ele in string:
      if string in punctuations:
        string = string.replace(ele, "")
    return string

  def remove_stopwords(tokens):
      stop_words = set(stopwords.words("english"))
      filtered_sentence = [w for w in tokens if not w in stop_words]
      #print(filtered_sentence)
      return filtered_sentence
  
  #removing all stop words  
  tokens_no_stop = remove_stopwords(tokens)
  #removing all punctuation
  tokens_no_stop_punc = [remove_punc(token) for token in tokens_no_stop]
  #list comp to remove any empty string
  filtered_tokens = [i for i in tokens_no_stop_punc if i]
  #print(mod_tokens_no_punc)
  return filtered_tokens


#words = make_lower_remove_punc(tokens)


def preprocess_data(wikidata_txt):
    sentences = sentence_segmentation(wikidata_txt)
    tokens = make_tokens(sentences)
    words = make_lower_remove_stopwords_punc(tokens)
    return words

if __name__ == "__main__":
    with open(r'C:/Users/Colmr/Desktop/UE803 - Data Science/project/webnlg-test.txt', encoding='utf-8') as infile:
        wikidata_txt = infile.read()
        
    print(preprocess_data(wikidata_txt))

        
