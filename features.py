# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:59:30 2022

@author: Dimitra
"""

# Import necessary libraries
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import numpy as np
import math
import collections
import textstat



#%%

####### Text Preprocessing #######

stop_words = set(stopwords.words('english'))

def preprocessing(sentences, text_type=None):
    '''
    -- Implements all necessary preprocessing, to prepare text for feature extraction --
      
     inputs: 
         - sentences (str): plain text split into sentences
         - text_type (str, default = None): controls the preprocessing steps to be taken
    returns:
          - text_tokens (list): a text read into a list, each element of the list is a preprocessed text's sentence       
          - text_pos (list containing lists of tuples): a text read into a list, each element of the list is another list 
                                                        consisting of tuples: ('preprocessed token', 'POS_tag')
    '''
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    text_tokens = []
    text_pos = [] 
    if text_type == 'title':
        sentences = list([sentences])      
    for sentence in sentences:     
        tokens = tokenizer.tokenize(sentence)  
        stemmer = PorterStemmer()
        sentence_tokens = []
        sent_tags = []
        tokenized_sentence = ''
        for tok in tokens:
          if tok.lower() not in stop_words:
             if text_type == None or text_type == 'title':
                  tok = stemmer.stem(tok)
             sentence_tokens += [tok]
             if text_type == 'POS':
                 sent_tags = pos_tag(sentence_tokens)
             tokenized_sentence = ' '.join(sentence_tokens)
        text_pos += [sent_tags]
        text_tokens += [tokenized_sentence]
    if text_type == None or text_type == 'title':
        return text_tokens
    return text_pos


####### Functions for Feature Extraction #######

def get_POS_features(text):
    '''
    -- calculates the number of named entities and numerals normalized per number of tokens per sentence --    
       input:
            - text : a list containing lists of tuples with the pos tags of a text's sentences       
       returns:
           - ne_list: a list of named entity scores per sentence
           - num_list: a list of numeral scores per sentence 
    '''     
    coord_list = []
    num_list = []
    sub_lis =[]
    for sentence in text:
        counter = 0
        counter_num = 0
        counter_sub = 0
        for tup in sentence:
            if 'CC' in tup:
                counter+=1
            if 'CD' in tup:
                counter_num +=1
            if 'IN' in tup:
                counter_sub +=1
        if len(sentence) == 0:
            coord_list += [0.0]
            num_list += [0.0]
            sub_lis += [0.0]
        else:
            counter = counter / float(len(sentence))
            counter_num = counter_num / float(len(sentence))
            counter_sub = counter_sub / float(len(sentence))
            coord_list += [counter]
            num_list += [counter_num]
            sub_lis += [counter_sub]
            
    return coord_list, num_list, sub_lis



def readability_metrics(tokenized_sentence):
    flesch_scores = []
    dalechall_scores = []
    difficult_words_ratio = []
    for sentence in tokenized_sentence:
        flesch_scores += [textstat.flesch_reading_ease(sentence)]
        dalechall_scores += [textstat.dale_chall_readability_score(sentence)]
        difficult_words_ratio += [0 if textstat.lexicon_count(sentence, removepunct=True) == 0 else (float(textstat.difficult_words(sentence))/textstat.lexicon_count(sentence, removepunct=True))]
    flesch_scores = np.array(flesch_scores)
    dalechall_scores = np.array(dalechall_scores)
    np.seterr(invalid='ignore')
    result_array_flesch=(flesch_scores-np.min(flesch_scores))/np.ptp(flesch_scores)
    result_array_flesch = np.nan_to_num(result_array_flesch)
    result_array_dalechall=(dalechall_scores-np.min(dalechall_scores))/np.ptp(dalechall_scores)
    result_array_dalechall= np.nan_to_num(result_array_dalechall)
    return list(result_array_flesch), list(result_array_dalechall), difficult_words_ratio
   

def length_sentence(tokenized_sentences):
    '''
    -- counts the ratio of each sentence length (ie. no of tokens) in terms of the overall mean length --    
       input:
            - tokenized_sentences: a list of a text's tokenized sentences       
       returns:
           - length_counts: a list of sentence scores for the text            
    '''
    length_counts = []
    for sentence in tokenized_sentences:
        length_counts += [len(sentence)]
    mean_lenght= np.mean(np.array(length_counts)) 
    length_counts = [x/float(mean_lenght) for x in length_counts]
    return length_counts


def sentence_position(tokenized_sentences):
    '''
    -- each sentence of the text gets a score depending on how far it is from the 1st sentence, normalized by 
        text's number of sentences --    
       input:
            - tokenized_sentences: a list of a text's tokenized sentences       
       returns:
           - sentence_count: a list of sentence position's scores
    '''
    x = [*range(len(tokenized_sentences), 0, -1)]
    sentence_count = [position/float(len(tokenized_sentences)) for position in x]
    return sentence_count


def tfIsf(text):
    '''
    -- each sentence of the text gets a tf-isf score (term frequency - inverse sentence frequency) --    
       input:
            - text: a list of a text's tokenized sentences       
       returns:
           - tfisf_score: a list of tf-isf scores, a score per sentence           
    '''
    tfisf_score = []
    sentences_with_word = {}
    string = ' '.join(text).lower()
    unique_words = set(string.split(' '))
    for word in unique_words:
        counter = 0
        for sentence in text:
            if word in sentence:
                counter +=1 
                sentences_with_word[word] = counter
    total_number_of_sentences = len(text)
    for sentence in text:
        tfisf_list = []
        sentence = sentence.lower()
        counts = collections.Counter(list(sentence.split(' ')))
        total_number_of_words_in_sent = len(counts)
        for word in counts.keys():
            tf = counts[word]/total_number_of_words_in_sent
            try:
                idf = total_number_of_sentences/sentences_with_word[word]+1
            except KeyError:
                idf = total_number_of_sentences/1           
            tfisf = tf*math.log(idf)
            tfisf_list += [tfisf]
        score = np.mean(tfisf_list)
        tfisf_score += [score]
    return tfisf_score


def jaccard_distance(token_set_a, token_set_b):
    '''
    --  calculates jaccard distance of two sets --    
       inputs:
            - token_set_a: list of tokens a
            - token_set_b: list of tokens b
       returns:
           - distance: jaccard distance score
           
    '''
    distance = len(set(token_set_a).intersection(token_set_b)) / float(len(set(token_set_a).union(token_set_b)))
    return distance



def sent_to_sent_cohesion(tokenized_sentences):
    '''
    -- calculates the similarity score of each sentence to each of the rest in terms of jaccard distance
        and takes the mean score --    
       input:
            - tokenized_sentences: a list of a text's tokenized sentences       
       returns:
           - score_per_sentence: a list of similarity scores, a score per sentence           
    '''
    score_per_sentence = []    
    
    for sentence in tokenized_sentences:
        score_list = []
        for s in tokenized_sentences:
            score = 0
            if s != sentence:
                score = jaccard_distance(sentence, s)
            score_list += [score]
            mean_score = np.mean(score_list)
        score_per_sentence  += [mean_score]    
    return score_per_sentence


def dict_count(text):
    '''
    --  creates a frequency dictionary of a sentence's tokens --    
       input:
            - text: a tokenized sentence
       returns:
           - freq_dict: frequency dictionary           
    '''
    freq_dict = collections.Counter(list(text.split(' ')))
    return freq_dict

def cosine_similarity(text_a, text_b):
    '''
    --  calculates cosine similarity between two tokenized sentences --    
       inputs:
            - text_a: sentence a
            - text_b: sentence b
       returns:
           - cosine similarity score
    '''
    dict_a = dict_count(text_a.lower())
    dict_b = dict_count(text_b.lower())
    intersection = set(dict_a.keys()) & set(dict_b.keys())
    a_list = []
    b_list = []
    for word in intersection:
        a_list += [dict_a[word]]
        b_list += [dict_b[word]]
    dot_product = np.dot(np.array(a_list), np.array(b_list))

    sum1 = sum([dict_a[x] ** 2 for x in dict_a.keys()])
    sum2 = sum([dict_b[x] ** 2 for x in dict_b.keys()])
    
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if denominator == 0:
        return 0.0
    else:
        return float(dot_product) / denominator


        

def similarity_predictive_sentence(tokenized_sentences, tfisf_scores):
    '''
    -- calculates the similarity score of each sentence to the most predictive one (based on tf-isf scores)
        in terms of cosine similarity mesure --
        
       inputs:
            - tokenized_sentences: a list of a text's tokenized sentences 
            - tfisf_scores: a list of tf-isf scores, a score per sentence
                  
       returns:
           - similarity_scores: a list of similarity scores, a score per sentence           
    '''
    similarity_scores = []
    max_tfisf = max(tfisf_scores)
    predictive_index = tfisf_scores.index(max_tfisf)
    for sentence in tokenized_sentences:
        score = cosine_similarity(tokenized_sentences[predictive_index], sentence)
        similarity_scores += [score]
    return similarity_scores




    








