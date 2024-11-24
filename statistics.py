# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:10:29 2022

@author: dim_k
"""

import metrics


true_ones = metrics.extract_gold_one_one("gold_standard/one_one.txt")
true_merging = metrics.extract_gold_merging("gold_standard/merging.txt")
true_splitting = metrics.extract_gold_splitting("gold_standard/splitting.txt")
true_deletions = metrics.extract_gold_deletions("gold_standard/deletions.txt")
true_insertions = metrics.extract_gold_insertions("gold_standard/insertions.txt")

pred_insertions, pred_deletions, pred_one_one, pred_merging, pred_splitting = metrics.extract_predictions ('aligner_results/all-mpnet-base-v2/itermax_align_results', 'source', 'target')
#%%

# helper function

def find_matches(tuples):
    for x in tuples:
        if type(x) == tuple:
            result = [1]*len(tuples)
            prev = tuples[0]
            for index, t in enumerate(tuples[1:]):
                for value in t:
                    if value in prev:
                        result[index+1] = 2
                        break
                prev = t
            result = [ i for i in result if i!=2 ]
            return result
        else:
            return tuples
    
#%%

# frequency of operations in gold standard

deletion_instances = 0
for nested in true_deletions:
    deletion_instances = deletion_instances + len([x for x in nested if not isinstance(x, str)])
    
    
insertion_instances = 0
for nested in true_insertions:
    insertion_instances = insertion_instances + len([x for x in nested if not isinstance(x, str)])


one_one_instances = 0
for nested in true_ones:
    one_one_instances = one_one_instances + len([x for x in nested if not isinstance(x, str)])      

new_merging = [find_matches(item) for item in true_merging]

merging_instances = 0
for nested in new_merging:
    merging_instances = merging_instances + len([x for x in nested if not isinstance(x, str)])
    

new_splitting = [find_matches(item) for item in true_splitting]

splitting_instances = 0
for nested in new_splitting:
    splitting_instances = splitting_instances + len([x for x in nested if not isinstance(x, str)])


total_operations_instances = deletion_instances+ insertion_instances + one_one_instances + merging_instances+splitting_instances

deletion_frequency = deletion_instances/total_operations_instances*100
insertion_frequency =  insertion_instances/total_operations_instances*100
one_one_frequency = one_one_instances/total_operations_instances*100
merging_frequency = merging_instances/total_operations_instances*100
splitting_frequency = splitting_instances/total_operations_instances*100



print(f"deletion_instances: {deletion_instances}")
print(f"insertion_instances: {insertion_instances}")
print(f"one_one_instances: {one_one_instances}")
print(f"merging_instances: {merging_instances}")
print(f"splitting_instances: {splitting_instances}")

print(f"total_operations_instances: {total_operations_instances}")

print(f"deletion_frequency: {deletion_frequency}")
print(f"insertion_frequency: {insertion_frequency}")
print(f"one_one_frequency: {one_one_frequency}")
print(f"merging_frequency: {merging_frequency}")
print(f"splitting_frequency: {splitting_frequency}")

#%%

# frequency of operations in predicted


pred_deletion_instances = 0
for nested in pred_deletions:
    pred_deletion_instances = pred_deletion_instances + len([x for x in nested if not isinstance(x, str)])
    
    
pred_insertion_instances = 0
for nested in pred_insertions:
    pred_insertion_instances = pred_insertion_instances + len([x for x in nested if not isinstance(x, str)])


pred_one_one_instances = 0
for nested in pred_one_one:
    pred_one_one_instances = pred_one_one_instances + len([x for x in nested if not isinstance(x, str)])      

pred_new_merging = [find_matches(item) for item in pred_merging]

pred_merging_instances = 0
for nested in pred_new_merging:
    pred_merging_instances = pred_merging_instances + len([x for x in nested if not isinstance(x, str)])
    

pred_new_splitting = [find_matches(item) for item in pred_splitting]

pred_splitting_instances = 0
for nested in pred_new_splitting:
    pred_splitting_instances = pred_splitting_instances + len([x for x in nested if not isinstance(x, str)])


pred_total_operations_instances = pred_deletion_instances+ pred_insertion_instances + pred_one_one_instances + pred_merging_instances+pred_splitting_instances

pred_deletion_frequency = pred_deletion_instances/pred_total_operations_instances*100
pred_insertion_frequency =  pred_insertion_instances/pred_total_operations_instances*100
pred_one_one_frequency = pred_one_one_instances/pred_total_operations_instances*100
pred_merging_frequency = pred_merging_instances/pred_total_operations_instances*100
pred_splitting_frequency = pred_splitting_instances/pred_total_operations_instances*100



print(f"pred_deletion_instances: {pred_deletion_instances}")
print(f"pred_insertion_instances: {pred_insertion_instances}")
print(f"pred_one_one_instances: {pred_one_one_instances}")
print(f"pred_merging_instances: {pred_merging_instances}")
print(f"pred_splitting_instances: {pred_splitting_instances}")

print(f"pred_total_operations_instances: {pred_total_operations_instances}")

print(f"pred_deletion_frequency: {pred_deletion_frequency}")
print(f"pred_insertion_frequency: {pred_insertion_frequency}")
print(f"pred_one_one_frequency: {pred_one_one_frequency}")
print(f"pred_merging_frequency: {pred_merging_frequency}")
print(f"pred_splitting_frequency: {pred_splitting_frequency}")


