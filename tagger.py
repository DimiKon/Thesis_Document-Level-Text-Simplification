# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 12:42:45 2022

@author: dim_k
"""

import pickle
import csv
import re

#%%

def make_zeros(number):
    return [0] * number


def deletions_labels(src, aligns):
    
    sentence_id = []
    label = []
    
    tmp_i = []
    for i in (0, len(src)):
        tmp_i = make_zeros(i)
         
        
    for elem in aligns:
        tmp_i[elem[0]]+=1
           
    for k in range(len(tmp_i)):
        sentence_id += [k]
        label = [0 if tmp_i[k] == 0 else 1 for k in range (len(tmp_i))]
        
    return sentence_id, label

#%%

text_id = []
sentence_id = []
sentence = []
label = []

with open('itermax_align_results', 'rb') as rlts:
    align_itermax= pickle.load(rlts)
    with open('target', 'rb') as trg:
        target_text = pickle.load(trg)
        total_wordcount_target = 0
        total_target = 0
        for nested in target_text:
            total_target = total_target + len([x for x in nested])
            word_lists_target = [(re.findall(r'\w+', x)) for x in nested]
            total_wordcount_target = total_wordcount_target + sum([len(elem) for elem in word_lists_target])
        average_target_length = total_target/len(target_text)
        average_target_wordcount = total_wordcount_target/len(target_text)
        print(average_target_length)
        print(average_target_wordcount)
    with open('source', 'rb') as src:
        source_text = pickle.load(src)
        total_wordcount = 0
        total_source = 0
        for nested in source_text:
            total_source = total_source + len([x for x in nested])
            word_lists = [(re.findall(r'\w+', x)) for x in nested]
            total_wordcount = total_wordcount + sum([len(elem) for elem in word_lists])
        average_source_length = total_source/len(source_text)
        average_source_wordcount = total_wordcount/len(source_text)
        print(average_source_length)
        print(average_source_wordcount)
        for index, (i, t) in enumerate(zip(source_text, target_text)):
            text_id += [index]* len(i)
            text_src = [' '.join(source_text[i]) for i in text_id]
            text_trg = [' '.join(target_text[i]) for i in text_id]
        sentence = [item for sublist in source_text for item in sublist]
        for source, align in zip(source_text, align_itermax):
            ids, labels = deletions_labels(source, align)
            sentence_id += ids
            label += labels
#%%

rows = zip(text_id, text_src, text_trg, sentence_id, sentence, label)


with open('train.csv', "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
        
#%%
import pandas as pd

data = pd.read_csv("train.csv", engine='python')

print(data.head())
print(data.shape)

'''
deleted = (data[data["label"] == 0]).count()
not_deleted = (data[data["label"] == 1]).count()

print(deleted)
print(not_deleted)

'''
label_counts = data["label"].value_counts()
print(label_counts)


#perc = 100 * data.label.value_counts() / len(data.label)

perc = data["label"].value_counts(normalize=True)
print(perc)




