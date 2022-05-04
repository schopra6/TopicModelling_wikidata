# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:46:57 2022

@author: JORGE
@modified :SAHIL
"""

import wikipedia
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import wptools
import spacy
# Load English Model
nlp = spacy.load('en_core_web_sm')

#========= Getting the categories ======================================================================


categories=['Airports','Artists','Astronauts','Building','Astronomical_objects','City','Comics_characters',
            'Companies','Foods','Transport','Monuments_and_memorials','Politicians','Sports_teams','Sportspeople','Universities_and_colleges','Written_communication']
#-----DEFINE THE NUMBER OF ARTICLES PER CATEGORY ----------
k=200
n=5
#---DEFINING LISTS TO STORE THE DATA-----
articles_data=[]
articles_data_2=[]
articles_data_3=[]
articles_data_4=[]
articles_data_5=[]



#========= Getting the articles from each category =======================================================

sparql=SPARQLWrapper("http://dbpedia.org/sparql/")

# We define the general SPARQL query in order to search 'k' articles in a 'category'


for category in categories:
    try:
        query=f"""
        PREFIX dcterms:<http://purl.org/dc/terms/>
        PREFIX dbc:<http://dbpedia.org/resource/Category:>
        
        SELECT ?label WHERE {{
        ?label
        dcterms:subject/skos:broader*
        dbc:{category} .
        }}
        LIMIT {k}
        """
        
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results=sparql.query().convert()
        
        for result in results["results"]["bindings"]:
            article_c=[]
            #---GET THE ARTICLE NAME
            article_link=result["label"]["value"]
            article=article_link.replace('http://dbpedia.org/resource/','')
            article_c=[category,article]          
            articles_data.append(article_c)
            
    except:
        #print("Error")
        articles_wkp_list=wikipedia.search(category,k)
        for result_wkp in articles_wkp_list:
            article_c=[]
            article_c=[category,result_wkp]           
            articles_data.append(article_c)
            

#--------------------GET THE WIKIPEDIA PAGE CONTENT--------------------------------------
for i in range(0,len(articles_data)):
    content=''
    article_content=[]
    
    page=wptools.page(articles_data[i][1],silent=False)
    try:
        page.get_query()
    except:
        pass
    try:
        content=page.data['extract']
    except KeyError:
        pass
    nlp(content)
    article_content=[articles_data[i][0],articles_data[i][1],content]
    articles_data_3.append(article_content)
    
#---------------------GET THE INFOBOX-----------------------------------------------------      
for i in range(0,len(articles_data)):
    infobox=''
    article_infobox=[]
    
    page=wptools.page(articles_data[i][1],silent=False)
    try:
        page.get_parse()
    except:
        pass
    try:
        infobox=page.data['infobox']
    except KeyError:
        pass
    article_infobox=[articles_data_3[i][0],articles_data_3[i][1],articles_data_3[i][2],infobox]
    articles_data_4.append(article_infobox)

#---------------------GET WIKIDATA STATEMENTS-----------------------------------------------   
for i in range(0,len(articles_data)):
    statements=''
    article_statement=[]
    
    page=wptools.page(articles_data[i][1],silent=False)
    try:
        page.get_wikidata()
    except:
        pass
    #try:
    statements=page.data['wikidata']
    #except KeyError:
    #    pass
    article_statement=[articles_data_4[i][0],articles_data_4[i][1],articles_data_4[i][2],articles_data_4[i][3],statements]
    articles_data_5.append(article_statement)

#--------------------SAVE ALL THE DATA INTO A CSV FILE -----------------------------
articles_csv=pd.DataFrame(articles_data_5,columns=['category','article','page_content','infobox','statements'])
articles_csv[articles_csv['page_content'].apply(lambda x : sum(1 for dummy in nlp(x).sents)) > n]
articles_csv.to_csv('articles.csv', index=False, header=['category','article','page_content','infobox','statements'])
