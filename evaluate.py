# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:38:17 2022

@author: dim_k
"""

import baseline_initial
from rouge import FilesRouge
import pickle
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import D_sari
import numpy as np
import textstat

#%%

def write_predicted_simple(predictions, sentence_filepath, lengths_filepath, output_simple):
    '''
    -- gets the sentences which are the most likely to be part of a summary, based on the model's scores, 
       and writes the respective summaries in a file, one summary per each text of the test/validation dataset --
      
     inputs: 
         - predictions (list): the output array of the function "predict"
         - sentence_filepath (file path): a txt file where all sentences of a text are kept 
                                         (output of function "parse_newsroom_file")
         - lengths_filepath (file path): a pickle file with an array which keeps lengths of
                                         all texts in terms of sentences counts
                                         o(utput of function "parse_newsroom_file")
         - output_summaries (file path): the output txt file
    '''
    length_array = pickle.load(open(lengths_filepath, 'rb'))
           
    start = 0
    end = 0
    for i, length in enumerate(length_array):  
        with open(sentence_filepath, encoding = 'utf-8') as file:
            with open(output_simple, 'a', encoding = 'utf-8') as output:
                sent_to_score = {}
                start = end
                end = start + length_array[i]
                text_pred = predictions[start:end]
                text_sentences = file.readlines()[start:end]
                zip_iterator = zip(text_sentences, text_pred)
                sent_to_score = dict(zip_iterator)
                text_summary = [x[0].strip('\n') for x in sent_to_score.items() if x[1] == 1]
                if text_summary:
                    text_summary = ' '.join(text_summary)
                    output.write(text_summary + '\n')
                else:
                    text_summary.append('none')
                    text_summary = ' '.join(text_summary)
                    output.write(text_summary + '\n')

#%%


def avg_tokens_per_text(file):
    data = [[item.strip() for item in sent_tokenize(line)] for line in open(file, encoding = 'utf-8')]
    total_wordcount = 0
    total_source = 0
    for nested in data:
        total_source = total_source + len([x for x in nested])
        #print(total_source)
        word_lists = [(re.findall(r'\w+', x)) for x in nested]
        total_wordcount = total_wordcount + sum([len(elem) for elem in word_lists])
        #print(total_wordcount)
        average_source_length = total_source/len(data)
        average_source_wordcount = total_wordcount/len(data)
    return average_source_length, average_source_wordcount

#%%
def metric_rouge (pred_path, true_path):
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(pred_path, true_path, avg=True)
    return scores

#%%
def bleu_score(pred_path, true_path):
    
    with open(true_path, encoding = 'utf-8') as true:
        with open(pred_path, encoding = 'utf-8') as output:
            text_true = true.readlines()
            ref_text = output.readlines()
            tokenized_true = [word_tokenize(sent) for sent in text_true]
            tokenized_ref = [[word_tokenize(sent)] for sent in ref_text]
            #print(tokenized_ref)
            score = corpus_bleu(tokenized_ref, tokenized_true)
            return score
            
#%%

def D_SARI(source, true_simple, output_simple):
    
    counter = 0
    scores = []
    keep =[]
    delete =[]
    add =[]
    with open(true_simple, encoding = 'utf-8') as true:
        with open(output_simple, encoding = 'utf-8') as output:
            with open (source, encoding = 'utf-8' ) as source_text:
                text_true = true.readlines()
                text_output = output.readlines()
                text_source = source_text.readlines()
                for i in range(len(text_true)):
                    [finalscore, avgkeepscore, avgdelscore, avgaddscore] = D_sari.D_SARIsent(" ".join([text_source[i]]), " ".join([text_output[i]]), [text_true[i]])
                    scores += [finalscore]
                    keep += [avgkeepscore]
                    delete += [avgdelscore]
                    add += [avgaddscore]
                    counter += 1
                    #print(counter)
    scores = np.mean(np.array(scores))
    keep = np.mean(np.array(keep))
    delete = np.mean(np.array(delete))
    add = np.mean(np.array(add))
    return scores, keep, delete, add



#%%

def readability_FKGL(file):
    counter = 0
    FKGL_scores = []
    with open(file, encoding = 'utf-8') as f:
        text = f.readlines()
        for i in range(len(text)):
            score = textstat.flesch_reading_ease(" ".join([text[i]]))
            FKGL_scores +=[score]
            counter += 1
            #print(counter)
    FKGL_scores = np.mean(np.array(FKGL_scores))   
    return FKGL_scores                        

    
#%%
#print(readability_FKGL('pred.txt'))



#%%
#print(D_SARI('source.txt', 'true.txt', 'pred.txt'))

         
#%%
#print(bleu_score('pred.txt', 'true.txt'))

#bleu.corpus_bleu([[ref_a], [ref_b]], [hyp, hyp])
                    
#%%
############# Development SET ###############


#y_pred = baseline.predict('./dev/features', './train/model', './dev/labels')

#write_predicted_simple(y_pred, './dev/sentences', './dev/num_of_sentences_per_text', './dev/output_simple')

#print(metric_rouge('./dev/output_simple', './dev/true_simple'))


#print(bleu_score('./dev/output_simple', './dev/true_simple'))




#%%

############# Test SET ###############

#y_pred = baseline_initial.predict('./test/features', './train/model', './test/labels')

#write_predicted_simple(y_pred, './test/sentences', './test/num_of_sentences_per_text', './test_best_threshold/output_simple')

#print(metric_rouge('./test/output_simple', './test/true_simple'))

#print(avg_tokens_per_text('./test_best_threshold/output_simple'))

#print(bleu_score('./test_best_threshold/output_simple', './test/true_simple'))

#print(D_SARI('./test/source.txt', './test/true_simple', './test_best_threshold/output_simple'))

#print(readability_FKGL('./test/output_simple'))

#print(readability_FKGL('./test/source.txt'))


# QuestEval
# RSRS (BERT)
