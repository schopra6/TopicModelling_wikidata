# import modules
import argparse
import os
import string
import pandas as pd
# import spacy 
# from spacy import displacy
from pprint import pprint
import nltk
from nltk.corpus import stopwords


class Preprocessor:
    def __init__(self, isLowercase = True, isTokenize = True, isStop=True, isNums = True, isPunct=True, isPostagging=False, isNER=False):

          self.isLowercase = isLowercase
          self.isTokenize = isTokenize
          self.isNums = isNums
          self.isStop = isStop
          self.isPunct = isPunct
          self.isPostagging = isPostagging
          self.isNER = isNER
          self.stop_words = set(stopwords.words('english'))

    def transform(self, texts):
        if self.isLowercase:
            print(">>>>> LOWER CASING ...")
            texts = (list(map(self.lower_case, texts)))
        if self.isNums:
            print(">>>>> REMOVING NUMBERS ...")
            texts = (list(map(self.remove_nums, texts)))
        if self.isPunct:
            print(">>>>> REMOVING PUNCTUATION ...")
            texts = (list(map(self.remove_punct, texts)))
        if self.isTokenize:
            print(">>>>> TOKENIZING ...")
            tokens = (list(map(self.tokenize, texts)))
        if self.isStop:
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


#       #POS Taggig each token 
#     def postagging(tokens):
#         spacy_pos_tagged = [(token, token.tag_, token.pos_) for token in tokens]
#         return spacy_pos_tagged

#       #Apply N.E.R to improve classification algorithm
#     def name_entity_recognition(texts):
#         sentences = segment_sentences(texts)

#         for sentence in sentences:
#             print("NEs:", [ne for ne in sentence.ents])
#             displacy.render(sentence, style='ent', jupyter=True)
#         return 

def main(inputpath, outputpath):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')
    preprocessor = Preprocessor()
    if df['description'].isna().any():
        newdf = df.dropna()
        newdf.reset_index(drop=True, inplace=True)
    
    print('>>>>> Wikipage Text processing...')
    newdf['preprocessed_page_content'] = preprocessor.transform(newdf['page_content'])
    
    print(">>>>> Wikidata description processing ...")
    newdf['preprocessed_description'] = preprocessor.transform(newdf['description'])
    
    preprocessed_df = newdf[['category', 'page_content', 'preprocessed_page_content', 'description', 'preprocessed_description']]
    columns = ['Person', 'Wikipage', 'Preprocessed Wikipage', 'Description', 'Preprocessed Description']
    preprocessed_df.columns = columns
    #output as csv
    preprocessed_df.to_csv(outputpath, index=False)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument("--input", type=str, default='data/extracted_data.csv',
                        help="Please provide the pathname to the csv file obtained after data extraction")
    
    parser.add_argument("--output", type=str, default='data/preprocessed_data.csv',
                        help="Please provide an output path for the preprocessed data")
    
    args = parser.parse_args()
    main(args.input, args.output)

