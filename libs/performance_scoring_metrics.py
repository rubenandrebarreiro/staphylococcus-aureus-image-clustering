# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:42:09 2020

@author: rubenandrebarreiro
"""

# Import F Classifier,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as f_1_score
from sklearn.feature_selection import f_classif as f_1_score

# Import cluster.DBSCAN Sub-Module,
# from SciKit-Learn Python's Library,
# as dbscan
from sklearn.metrics import silhouette_score as skl_silhouette_score

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library,
# as k_means
from sklearn.metrics import adjusted_rand_score as skl_adjusted_rand_score


def f_1_measure_score(xs_features, ys_labels_true):
    
    f_values, probabilities_f_values = f_1_score(xs_features, ys_labels_true) 
    
    
    return f_values, probabilities_f_values


def silhouette_score(xs_features, ys_labels_predicted):
    
    silhouette_score_average = skl_silhouette_score(xs_features, ys_labels_predicted)
    
    
    return silhouette_score_average

    
    