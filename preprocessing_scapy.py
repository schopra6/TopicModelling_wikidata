import pandas as pd
import spacy 
from spacy import displacy
from pprint import pprint
nlp = spacy.load('en_core_web_sm', entity=True)

#sentence segmentation
def segment_sentences(sp_wikidata):
  sentences = []
  for sentence in wikidata_txt.sents:
      sentences.append(sentence)
  return sentences

#tokenizing each word
def make_tokens(sp_wikidata):
    tokens = []
    for word in sp_wikidata:
        tokens.append(word)
    return tokens


#remove stop words & punctuation & lowercase
def remove_stop_punct(tokens):
  no_punct_tokens = [token for token in tokens if token.is_punct != True]
  moded_tokens = [token for token in no_punct_tokens if token.is_stop != True]
  return moded_tokens

  '''
    mod_tokens = [token for token in tokens if (token.is_stop != True or token.is_punct != True)]
    return [token.lower() for token in mod_tokens]
  '''
    
def preprocess(sp_wikidata):
    tokens = make_tokens(sp_wikidata)
    mod_tokens = remove_stop_punct(tokens)
    return mod_tokens

#POS Taggig each token 
def postagging(tokens):
  spacy_pos_tagged = [(token, token.tag_, token.pos_) for token in tokens]
  return spacy_pos_tagged

#Apply N.E.R to improve classification algorithm
def name_entity_recognition(sp_wikidata):
  sentences = segment_sentences(sp_wikidata)

  for sentence in sentences:
    print("NEs:", [ne for ne in sentence.ents])
    displacy.render(sentence, style='ent', jupyter=True)
  return 


if __name__ == "__main__":
    with open(r'/content/webnlg-test.txt', encoding='utf-8') as infile:
        wikidata_txt = infile.read()
    
    sp_wikidata = nlp(wikidata_txt) #apply spacy model to text
    tokens = preprocess(sp_wikidata)
    print(tokens)
    
    df = pd.DataFrame(tokens, columns =['Person', 'Wikipedia page text', 'Wikipedia page text after preprocessing', 
                                        'Wikidata description', 'Wikidata description after preprocessing'])
  
