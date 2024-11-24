# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:44:20 2022

@author: Dimitra
"""

import pickle
import re
#%%

# helper function 

def make_zeros(number):
    return [0] * number


# function to get predictions per simplification modification

def results(src, trg, aligns):
    insertion = []
    deletion = []
    one_one = []
    compression = []
    splitting =[]
    
    tmp_i = []
    for i in (0, len(src)):
        tmp_i = make_zeros(i)
        
    tmp_j = []
    for j in (0,len(trg)):
        tmp_j= make_zeros(j)  
        
    for elem in aligns:
        tmp_i[elem[0]]+=1
        tmp_j[elem[1]]+=1
        
    for k in range(len(tmp_i)):
        if tmp_i[k] == 0:
            deletion.append(k)        
        if tmp_i[k] > 1:
            for elem in aligns:
                if k == elem[0]:
                   splitting.append(elem)
    
    for k in range(len(tmp_j)):
          if tmp_j[k] == 0:
              insertion.append(k)
          if tmp_j[k] > 1:
               for elem in aligns:
                   if k == elem[1]:
                       compression.append(elem)
              
    for k in range(len(tmp_i)):
        if tmp_i[k] == 1:
            for elem in aligns:
                if k == elem[0]:
                    if elem not in compression:
                        one_one.append(elem)
    
            
    if len(insertion) == 0:
        insertion+=['none']
    
    if len(deletion) == 0:
        deletion+=['none']
        
    if len(one_one) == 0:
        one_one+=['none']
        
    if len(compression) == 0:
        compression += ['none']
        
    if len(splitting) == 0:
        splitting += ['none']
        
    return insertion, deletion, one_one, compression, splitting

#%%

# function to extract predictions

def extract_predictions (results_file, source, target):

    class_insertions = []
    class_deletions = []
    class_one_one_aligns = []
    class_merging = []
    class_splitting = []

    with open(results_file, 'rb') as rlts:
        align_itermax= pickle.load(rlts)
        with open(source, 'rb') as src:
            source_text = pickle.load(src)
            with open(target, 'rb') as trg:
                target_text = pickle.load(trg)
                for source, target, align in zip(source_text, target_text, align_itermax):
                    cl1, cl2, cl3, cl4, cl5 = results(source, target, align)
                    class_insertions += [cl1]
                    class_deletions += [cl2]
                    class_one_one_aligns += [cl3]
                    class_merging += [cl4]
                    class_splitting += [cl5]
                    
    return class_insertions, class_deletions, class_one_one_aligns, class_merging, class_splitting
                    
#%%

# functions to extract gold standard alignments

def extract_gold_one_one(file):
    true_one_one = []
    pattern='\((\d+, \d+)\)'
    empty = '[]'
    data = [line.strip() for line in open(file, 'r')]
    for line in data:
        if line == empty:
            true_one_one+=[["none"]]
        else:
            full = re.findall(pattern,line)
            data_1=[]
            for item in full:
                data_1.append(tuple(map(lambda x:int(x),item.split(','))))
            if data_1:
                true_one_one+=[data_1]
    return true_one_one

                
    
#%%

def extract_gold_merging(file):
    true_merging = []
    pattern='\((\d+, \d+)\)'
    empty = '[]'
    data = [line.strip() for line in open(file, 'r')]
    for line in data:
        if line == empty:
            true_merging+=[["none"]]
        else:
            full = re.findall(pattern,line)
            data_1=[]
            for item in full:
                data_1.append(tuple(map(lambda x:int(x),item.split(','))))
            if data_1:
                true_merging+=[data_1]
            
    return true_merging



#%%

def extract_gold_splitting(file):
    true_splitting = []
    pattern='\((\d+, \d+)\)'
    empty = '[]'
    data = [line.strip() for line in open(file, 'r')]
    for line in data:
        if line == empty:
            true_splitting+=[["none"]]
        else:
            full = re.findall(pattern,line)
            data_1=[]
            for item in full:
                data_1.append(tuple(map(lambda x:int(x),item.split(','))))
            if data_1:
                true_splitting+=[data_1]
            
    return true_splitting



#%%

def extract_gold_deletions(file):
    true_deletions = []
    pattern='\d+(?:,\d+)?'
    empty = '[]'
    data = [line.strip() for line in open(file, 'r')]
    for line in data:
        if line == empty:
            true_deletions+=[["none"]]
        else:
            full = re.findall(pattern,line)
            data_1=[]
            for item in full:
                data_1.append([int(x) for x in item.split(',')])
            if data_1:
                flat_list = [x for xs in data_1 for x in xs]
                true_deletions+=[flat_list]
    return true_deletions


#%%


def extract_gold_insertions(file):
    true_insertions = []
    pattern='\d+(?:,\d+)?'
    empty = '[]'
    data = [line.strip() for line in open(file, 'r')]
    for line in data:
        if line == empty:
            true_insertions+=[["none"]]
        else:
            full = re.findall(pattern,line)
            data_1=[]
            for item in full:
                data_1.append([int(x) for x in item.split(',')])
            if data_1:
                flat_list = [x for xs in data_1 for x in xs]
                true_insertions+=[flat_list]
    
    return true_insertions



#%%

# helper functions

def common_elements(list1, list2):
    return len([element for element in list1 if element in list2])

def prec(correct, all_pred):
    return correct/all_pred*100

def rec(correct, all_gold):
    return correct/all_gold*100



#%%

# METHOD 1: micro-scores per category
# function to calculate micro-precision, micro-recall, micro-f1


def metrics_micro (true_list, pred_list):
    all_gold_alignments = 0
    all_extracted_alignments = 0
    correct_extracted_alignments = 0
    zipped = zip(pred_list, true_list)
    for pair in zipped:
        correct_extracted_alignments+= common_elements(pair[0], pair[1])
    for true in true_list:
        all_gold_alignments+=len(true)
    for pred in pred_list:
         all_extracted_alignments+=len(pred)
    precision = correct_extracted_alignments/all_extracted_alignments*100
    recall = correct_extracted_alignments/all_gold_alignments*100
    f1 = 2*(precision*recall)/(precision+recall)
    
    
    
    ### Print to check results ###
    
    #print(all_gold_alignments)
    #print(all_extracted_alignments)
    #print(correct_extracted_alignments)
    
    return precision, recall, f1



#%%

# METHOD 1: micro-scores in total
# function to calculate micro-precision, micro-recall, micro-f1


def metrics_micro_all (true_list, pred_list):
    all_gold_alignments = 0
    all_extracted_alignments = 0
    correct_extracted_alignments = 0
    zipped = zip(pred_list, true_list)
    for pair in zipped:
        z = zip(pair[0], pair[1])
        for p in z: 
            correct_extracted_alignments += common_elements(p[0], p[1])
            all_extracted_alignments += len(p[0])
            all_gold_alignments += len(p[1])
            
            ### Print to check results ###
            
            #print(all_gold_alignments)
            #print(all_extracted_alignments)
            #print(correct_extracted_alignments)
            
            
    precision = correct_extracted_alignments/all_extracted_alignments*100
    recall = correct_extracted_alignments/all_gold_alignments*100
    f1 = 2*(precision*recall)/(precision+recall)
    
    return precision, recall, f1


#%%

# METHOD 2: macro-scores per category
# function to calculate macro-precision, macro-recall, macro-averaging


def metrics_macro (true_list, pred_list):
    total_prec = 0
    total_rec = 0
    zipped = zip(pred_list, true_list)
    for pair in zipped:
        correct_extracted_alignments= common_elements(pair[0], pair[1])
        all_gold_alignments = len(pair[1])
        all_extracted_alignments = len(pair[0])
        total_prec += prec(correct_extracted_alignments,all_extracted_alignments)
        total_rec += rec(correct_extracted_alignments, all_gold_alignments)
    
        
        ### Print to check results ###
        
        
        #print(all_gold_alignmements)
        #print(all_extracted_alignments)
        #print(correct_extracted_alignments)
        
    
    
    precision = total_prec/len(pred_list)
    recall = total_rec/len(pred_list)
    
    f1 = 2*(precision*recall)/(precision+recall)
    
    return precision, recall, f1


#%%


# METHOD 2: macro-scores in total
# function to calculate macro-precision, macro-recall, macro-averaging


def metrics_macro_all (true_list, pred_list):
    total_prec = 0
    total_rec = 0
    zipped = zip(pred_list, true_list)
    c = 0
    for pair in zipped:
        z = zip(pair[0], pair[1])
        for p in z:
            correct_extracted_alignments = common_elements(p[0], p[1])
            all_extracted_alignments = len(p[0])
            all_gold_alignments = len(p[1])
            total_prec += prec(correct_extracted_alignments,all_extracted_alignments)
            total_rec += rec(correct_extracted_alignments, all_gold_alignments)
            c+=1
            
            ### Print to check results ###
            
            #print(all_gold_alignments)
            #print(all_extracted_alignments)
            #print(correct_extracted_alignments)
        
    precision = total_prec/c
    recall = total_rec/c
    
    f1 = 2*(precision*recall)/(precision+recall)
    
    return precision, recall, f1

#%%


# METHOD 3: weighted-macro-scores per category
# function to calculate weighted macro-precision, weighted macro-recall, weighted macro-averaging


def metrics_macro_weighted (true_list, pred_list):

    total_prec = 0
    total_rec = 0
    all_gold = 0

    zipped = zip(pred_list, true_list)
    for pair in zipped:
        correct_extracted_alignments= common_elements(pair[0], pair[1])
        all_gold_alignments = len(pair[1])
        all_gold += all_gold_alignments
        all_extracted_alignments = len(pair[0])
        total_prec += prec(correct_extracted_alignments,all_extracted_alignments)*all_gold_alignments
        total_rec += rec(correct_extracted_alignments, all_gold_alignments)*all_gold_alignments
        
        ### Print to check results ###
            
        #print(all_gold_alignments)
        #print(all_extracted_alignments)
        #print(correct_extracted_alignments)
    
    
    precision = total_prec/all_gold
    recall = total_rec/all_gold
    
    f1 = 2*(precision*recall)/(precision+recall)
    
    return precision, recall, f1


# METHOD 3: weighted-macro-scoress in total
# function to calculate weighted macro-precision, weighted macro-recall, weighted macro-averaging

#%%
def metrics_macro_weighted_all (true_list, pred_list):
    total_prec = 0
    total_rec = 0
    zipped = zip(pred_list, true_list)
    gold_alignments = 0
    for pair in zipped:
        z = zip(pair[0], pair[1])
        for p in z:
            correct_extracted_alignments = common_elements(p[0], p[1])
            all_extracted_alignments = len(p[0])
            all_gold_alignments = len(p[1])
            total_prec += prec(correct_extracted_alignments,all_extracted_alignments)*all_gold_alignments
            total_rec += rec(correct_extracted_alignments, all_gold_alignments)*all_gold_alignments
            gold_alignments+=all_gold_alignments
            
            ### Print to check results ###
            
            #print(all_gold_alignments)
            #print(all_extracted_alignments)
            #print(correct_extracted_alignments)
        
    precision = total_prec/gold_alignments
    recall = total_rec/gold_alignments
    
    f1 = 2*(precision*recall)/(precision+recall)
    
    return precision, recall, f1