# import modules
import argparse
import os
import string
import pandas as pd
import spacy 
from spacy import displacy
from pprint import pprint
import nltk
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self, lowercase = True, isTokenize = True, no_stop=True, no_nums = True,no_punct=True, postagging=False, name_entity_recognition=False):

          self.lowercase = lowercase
          self.isTokenize = isTokenize
          self.no_nums = no_nums
          self.no_stop = no_stop
          self.no_punct = no_punct
          self.postagging = postagging
          self.name_entity_recognition = name_entity_recognition
          self.stop_words = set(stopwords.words('english'))
          #self.nlp = spacy.load('en_core_web_sm')

    def transform(self, texts):
        if self.lowercase:
            print(">>>>> LOWER CASING ...")
            texts = (list(map(self.lower_case, texts)))
        if self.no_nums:
            print(">>>>> REMOVING NUMBERS ...")
            texts = (list(map(self.remove_nums, texts)))
        if self.no_punct:
            print(">>>>> REMOVING PUNCTUATION ...")
            texts = (list(map(self.remove_punct, texts)))
        if self.isTokenize:
            print(">>>>> TOKENIZING ...")
            tokens = (list(map(self.tokenize, texts)))
        if self.no_stop:
            print(">>>>> REMOVING STOP WORDS ...")
            tokens = (list(map(self.remove_stop, tokens)))  
            preprocessed_texts = [" ".join(token) for token in tokens]
        print("TRANSFORMATION COMPLETE !")
        return preprocessed_texts

            #sentence segmentation
    def segment_sentences(text):
        sentences = []
        for sentence in text:
            sentences.append(sentence)
        return sentences

    def tokenize(self, text):
        return nltk.word_tokenize(text)
    
    
    def lower_case(self, text):
        return text.lower()
    
    def remove_nums(self, text):
        nums_translator = str.maketrans('', '', '0123456789')
        return text.translate(nums_translator)

      #remove stop words & punctuation & lowercase
    def remove_stop(self, tokens):
        return  [token for token in tokens if token not in self.stop_words]


    def remove_punct(self, text):
        punct_translator = str.maketrans('', '', string.punctuation)
        return text.translate(punct_translator)


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

def main(inputpath, outputpath):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')
    preprocessor = Preprocessor()
    if df['description'].isna().any():
        newdf = df.dropna()
        newdf.reset_index(drop=True, inplace=True)
    
    print('▶ Wikipage Text processing...')
    newdf['preprocessed_page_content'] = preprocessor.transform(newdf['page_content'])
    
    print(">>>>> Wikidata description processing ...")
    newdf['preprocessed_description'] = preprocessor.transform(newdf['description'])
    
    preprocessed_df = pd.DataFrame(columns=['Person', 'Wikipage', 'Preprocessed Wikipage', 'Description', 'Preprocessed Description'])
    columns = ['Person', 'Wikipage', 'Preprocessed Wikipage', 'Description', 'Preprocessed Description']
    preprocessed_df.columns = columns
    #output as csv
    preprocessed_df.to_csv(outputpath, index=False)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument("--input", type=str, default='data/data.csv',
                        help="pathname to the input file for preprocessing (a csv file obtained after running the extraction script)")
    
    parser.add_argument("--output", type=str, default='data/preprocessed_data.csv',
                        help="desired path to the output csv file")
    
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, args.verbose)
