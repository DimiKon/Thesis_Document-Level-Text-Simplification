# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:02:20 2022

@author: Dimitra
"""

#%%
# import libraries

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
try:
    import networkx as nx
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
except ImportError:
    nx = None

import codecs
import regex as re

from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize

import pickle

#%%

class TextAligner(object):
    def __init__(self,  model = 'all-mpnet-base-v2', distortion = 0.0, matching_methods = "mi"):
        # different sentence embeddings models to experiment with
        model_names = {
            'all-mpnet-base-v2': (SentenceTransformer),
            "all-MiniLM-L12-v2": (SentenceTransformer),
            "all-distilroberta-v1":(SentenceTransformer),
            "all-MiniLM-L6-v2": (SentenceTransformer)}
           
        all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}
        
        
        self.model = model
        self.emb_model = None
        if model in model_names:
            model_class = model_names[model]
            self.emb_model = model_class(model)
        self.distortion = distortion
        self.matching_methods = [all_matching_methods[m] for m in matching_methods]

     
    # Embeddings Matrix   
    def get_2d_embed_matrix (self, model, text):
        if isinstance(text, str):
            sentence_list = sent_tokenize(text)
        sentence_list = sent_tokenize(text[0])
        embeddings = model.encode(sentence_list)
        return embeddings

    # Cosine Similarity Matrix
    @staticmethod
    def get_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (cosine_similarity(X, Y) + 1.0) / 2.0


    # Alignment matrix (0-1s)
    @staticmethod
    def get_alignment_matrix(sim_matrix: np.ndarray):
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        return forward, backward.transpose()

    # Probably not useful method for contextualized embeddings
    @staticmethod
    def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix

        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

        return np.multiply(sim_matrix, distortion_mask)
    
    
    # GRAPH METHOD_mwmf
    @staticmethod
    def get_max_weight_match(sim: np.ndarray) -> np.ndarray:
        if nx is None:
            raise ValueError("networkx must be installed to use match algorithm.")
        def permute(edge):
            if edge[0] < sim.shape[0]:
                return edge[0], edge[1] - sim.shape[0]
            else:
                return edge[1], edge[0] - sim.shape[0]
        G = from_biadjacency_matrix(csr_matrix(sim))
        matching = nx.max_weight_matching(G, maxcardinality=True)
        matching = [permute(x) for x in matching]
        matching = sorted(matching, key=lambda x: x[0])
        res_matrix = np.zeros_like(sim)
        for edge in matching:
            res_matrix[edge[0], edge[1]] = 1
        return res_matrix

    # ITERMAX method
    @staticmethod
    def iter_max(sim_matrix: np.ndarray, max_count: int=2) -> np.ndarray:
        alpha_ratio = 0.9
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        inter = forward * backward.transpose()

        if min(m, n) <= 2:
            return inter

        new_inter = np.zeros((m, n))
        count = 1
        while count < max_count:
            mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
            mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
            mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
            mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
            if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
                mask *= 0.0
                mask_zeros *= 0.0

            new_sim = sim_matrix * mask
            fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
            bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
            new_inter = fwd * bac

            if np.array_equal(inter + new_inter, inter):
                break
            inter = inter + new_inter
            count += 1
        return inter
    
    

    def get_sent_aligns(self, src_text, trg_text):
        
        src_embed = self.get_2d_embed_matrix(self.emb_model, src_text)
        trg_embed = self.get_2d_embed_matrix(self.emb_model, trg_text)
              
        all_mats = {}
        sim = self.get_similarity_matrix(src_embed, trg_embed)
        sim = self.apply_distortion(sim, self.distortion)

        all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
        all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
        if "mwmf" in self.matching_methods:
            all_mats["mwmf"] = self.get_max_weight_match(sim)
        if "itermax" in self.matching_methods:
            all_mats["itermax"] = self.iter_max(sim)

        aligns = {x: set() for x in self.matching_methods}
        for i in range(len(src_embed)):
            for j in range(len(trg_embed)):
                for ext in self.matching_methods:
                    if all_mats[ext][i, j] > 0:
                        aligns[ext].add((i, j))
        for ext in aligns:
            aligns[ext] = sorted(aligns[ext])
        return aligns
    

#%%


my_align = TextAligner('all-mpnet-base-v2')


files = ['sample_200_source', 'sample_200_target']

max_sent_id = None

original_corpora = []

for file in files:
    corpus = [re.sub(r'(?<=[.])(?=[^\s])', r' ', l) for l in codecs.open(file, 'r', 'utf-8').readlines()]
    original_corpora.append(corpus[:max_sent_id])



align_results_itermax = []
align_results_graph = []


source =[]
target =[]


for text_id in range(len(original_corpora[0])):
    source_text = [original_corpora[0][text_id]]
    sentence_list_source = sent_tokenize(source_text[0])
    source+= [sentence_list_source]
    target_text = [original_corpora[1][text_id]]
    sentence_list_target = sent_tokenize(target_text[0])
    target+= [sentence_list_target]  
    x = my_align.get_sent_aligns(source_text, target_text)
    align_results_itermax.append(x['itermax'])
    align_results_graph.append(x['mwmf'])
    
pickle.dump(align_results_itermax, open('itermax_align_results', 'wb'))
pickle.dump(align_results_graph, open('graph_align_results', 'wb'))
pickle.dump(source, open('source', 'wb'))
pickle.dump(target, open('target', 'wb'))
