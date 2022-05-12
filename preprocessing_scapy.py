import argparse
import os
import string

import pandas as pd
import spacy 
from spacy import displacy
from pprint import pprint
nlp = spacy.load('en_core_web_sm', entity=True)

class Preprocessor:
    def __init__(self, lowercase = True, no_stop=True, no_nums = True,no_punct=True, postagging=False, name_entity_recognition=False, verbose=True):

          self.lowercase = lowercase
          self.no_nums = no_nums
          self.no_stop = no_stop
          self.no_punct = no_punct
          self.postagging = postagging
          self.name_entity_recognition = name_entity_recognition
          self.verbose = verbose
          self.nlp = spacy.load('en_core_web_sm')

    def transform(self, texts):
        if self.lowercase:
            if self.verbose:
                print(">>>>> LOWER CASING ...")
            texts = (list(map(self.lower_case, texts)))
        if self.no_nums:
            if self.verbose:
                print(">>>>> REMOVING NUMBERS ...")
            texts = (list(map(self.remove_nums, texts)))
        if self.verbose:
            texts =" ".join(map(str,texts))
            texts = self.nlp(texts)
            print(">>>>> TOKENIZING ...")
            tokens = self.tokenize(texts)
        if self.no_stop:
            if self.verbose:
                print(">>>>> REMOVING STOP WORDS ...")
            tokens = self.remove_stop(tokens)  
        if self.no_punct:
            if self.verbose:
                print(">>>>> REMOVING PUNCTUATION ...")
            tokens = self.remove_punct(tokens)
        return tokens

      #sentence segmentation
    def segment_sentences(text):
        sentences = []
        for sentence in text:
            sentences.append(sentence)
        return sentences

      #tokenizing each word
    def tokenize(self, text):
        tokens = []
        for word in text:
            tokens.append(word)
        return tokens

    
    def lower_case(self, text):
        return text.lower()
    
    def remove_nums(self, text):
        nums_translator = str.maketrans('', '', '0123456789')
        return text.translate(nums_translator)

      #remove stop words & punctuation & lowercase
    def remove_stop(self, tokens):
        return  [token for token in tokens if token.is_stop != True]

    def remove_punct(self, tokens):
        return [token for token in tokens if token.is_punct != True]


      #POS Taggig each token 
    def postagging(tokens):
        spacy_pos_tagged = [(token, token.tag_, token.pos_) for token in tokens]
        return spacy_pos_tagged

      #Apply N.E.R to improve classification algorithm
    def name_entity_recognition(texts):
        sentences = segment_sentences(texts)

        for sentence in sentences:
            print("NEs:", [ne for ne in sentence.ents])
            displacy.render(sentence, style='ent', jupyter=True)
        return 

def main(inputpath: str, outputpath: str, verbose: bool):
    """
    Function that starts after calling the script
    :param inputpath: path to the csv with the data for preprocessing
    :param outputpath: desired path to the output csv file
    :param verbose:  if True, the main steps will be printed during the execution
    """
     if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')
    preprocessor = Preprocessor(verbose=args.verbose)
    if verbose:
        print('▶ Wikipage Text processing...')
    newdf['Wikitext preprocessed'] = preprocessor.transform(df['article'])
    if verbose:
        print(">>>>> Wikidata description processing ...")
    newdf['Wikidescription preprocessed'] = preprocessor.transform(df['description'])
    
    #Add new dataFrame containing columns   
    #newdf = df[['Person', 'Wikitext', 'Wikitext preprocessed', 'Wikidescription', 'Wikidescription preprocessed']]  
    '''
    New DataFrame: 
                • column 1: person
                • column 2: Wikipedia page text
                • column 3: Wikipedia page text after preprocessing
                • column 4: Wikidata description
                • column 5: Wikidata description after preprocessing

    '''
    #outputs the processed data as a csv to the outputpath specified
    newdf.to_csv(outputpath, index=False)  
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument("--input", type=str, default='data/data.csv',
                        help="pathname to the input file for preprocessing (a csv file obtained after running the extraction script)")
    
    parser.add_argument("--output", type=str, default='data/preprocessed_data.csv',
                        help="desired path to the output csv file")
    
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, args.verbose)
