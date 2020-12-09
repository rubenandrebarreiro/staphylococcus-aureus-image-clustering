# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:42:09 2020

@author: rubenandrebarreiro
"""

# Import cluster.DBSCAN Sub-Module,
# from SciKit-Learn Python's Library,
# as dbscan
from sklearn.metrics import silhouette_score as skl_silhouette_score

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library,
# as k_means
from sklearn.metrics import adjusted_rand_score as skl_adjusted_rand_score

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import combinations,
# From the Iteration Tools' Python Library,
# as iteration_combinations
from itertools import combinations as iteration_combinations


def compute_confusion_matrix_rand_index_clustering(clustering_algorithm, ys_labels_true, ys_labels_predicted, num_clusters, final_clustering = False):
    
    num_examples = len(ys_labels_true)
    
    confusion_matrix_rand_index_clustering = matrix_array_zeros((2, 2))
    
    num_true_positives = 0
    num_false_positives = 0
    
    num_false_negatives = 0
    num_true_negatives = 0
    
    
    examples_pairs = list( iteration_combinations( range( 0, num_examples ), r = 2 ) )
    
    num_examples_pairs = len(examples_pairs)
    
    
    for example_pair in examples_pairs:
        
        example_pair_first_index = example_pair[0]
        example_pair_second_index = example_pair[1]
        
        # The Example Pair are in the same Group (same True Label)
        if(ys_labels_true[example_pair_first_index] == ys_labels_true[example_pair_second_index]):

            # The Example Pair are in the same Cluster (same Predicted Label)
            if(ys_labels_predicted[example_pair_first_index] == ys_labels_predicted[example_pair_second_index]):    
                num_true_positives = ( num_true_positives + 1 )
            
            # The Example Pair are in different Clusters (different Predicted Labels)
            else:    
                num_false_negatives = ( num_false_negatives + 1 )
        
        # The Example Pair are in different Groups (different True Labels)
        else:
            
            # The Example Pair are in the same Cluster (same Predicted Label)
            if(ys_labels_predicted[example_pair_first_index] == ys_labels_predicted[example_pair_second_index]):    
                num_false_positives = ( num_false_positives + 1 )
            
            # The Example Pair are in different Clusters (different Predicted Labels)
            else:    
                num_true_negatives = ( num_true_negatives + 1 )        
    
    
    confusion_matrix_rand_index_clustering[0][0] = int(num_true_positives)
    confusion_matrix_rand_index_clustering[0][1] = int(num_false_positives)
    
    confusion_matrix_rand_index_clustering[1][0] = int(num_false_negatives)
    confusion_matrix_rand_index_clustering[1][1] = int(num_true_negatives)
        
    
    return confusion_matrix_rand_index_clustering, num_examples_pairs


def compute_silhouette_score(xs_features, ys_labels_predicted):
    
    silhouette_score_average = skl_silhouette_score(xs_features, ys_labels_predicted)
    
    
    return silhouette_score_average

    
def compute_precision_score(num_true_positives, num_false_positives):
    
    precision_score = ( num_true_positives / ( num_true_positives + num_false_positives ) )
    
    
    return precision_score


def compute_recall_score(num_true_positives, num_false_negatives):
    
    recall_score = ( num_true_positives / ( num_true_positives + num_false_negatives ) )
    
    
    return recall_score


def compute_rand_index_score(num_true_positives, num_true_negatives, num_examples_pairs):
    
    rand_index_score = ( num_true_positives + num_true_negatives ) / num_examples_pairs
    
    
    return rand_index_score


def compute_f1_score(precision_score, recall_score):
    
    f1_score = ( 2 * ( ( precision_score * recall_score ) / ( precision_score + recall_score ) ) )
    
    
    return f1_score


def compute_adjusted_rand_score(ys_labels_true, ys_labels_predicted):
    
    adjusted_rand_score = skl_adjusted_rand_score(ys_labels_true, ys_labels_predicted)
    
    
    return adjusted_rand_score



def compute_clustering_performance_metrics(clustering_algorithm, xs_features, ys_labels_true, ys_labels_predicted, num_clusters, final_clustering = False):
        
    confusion_matrix_rand_index_clustering, num_examples_pairs = compute_confusion_matrix_rand_index_clustering(clustering_algorithm, ys_labels_true, ys_labels_predicted, num_clusters, final_clustering)
    
    
    num_true_positives = confusion_matrix_rand_index_clustering[0][0] 
    num_false_positives = confusion_matrix_rand_index_clustering[0][1]
    
    num_false_negatives = confusion_matrix_rand_index_clustering[1][0]
    num_true_negatives = confusion_matrix_rand_index_clustering[1][1]
    
    
    silhouette_score = compute_silhouette_score(xs_features, ys_labels_predicted)
    precision_score = compute_precision_score(num_true_positives, num_false_positives)
    recall_score = compute_recall_score(num_true_positives, num_false_negatives)
    rand_index_score = compute_rand_index_score(num_true_positives, num_true_negatives, num_examples_pairs)
    f1_score = compute_f1_score(precision_score, recall_score)
    adjusted_rand_score = compute_adjusted_rand_score(ys_labels_true, ys_labels_predicted)
    
    
    return silhouette_score, precision_score, recall_score, rand_index_score, f1_score, adjusted_rand_score, confusion_matrix_rand_index_clustering


def print_clustering_performance_metrics(clustering_algorithm, num_total_clusters, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores):
    
    print("\n\n")
    
    for current_num_clusters in range(0, ( num_total_clusters - 1 ) ):
        
        print("\n")
        
        print( "Performance Metrics for {} Clustering, with K = {} Cluster(s):".format( clustering_algorithm, ( current_num_clusters + 2 ) ) )
        
        print( " - Silhouette Score: {}".format(clusters_silhouette_scores[current_num_clusters]) )
        print( " - Precision Score: {}".format(clusters_precision_scores[current_num_clusters]) )
        print( " - Recall Score: {}".format(clusters_recall_scores[current_num_clusters]) )
        print( " - Rand Index Score: {}".format(clusters_rand_index_scores[current_num_clusters]) )
        print( " - F1 Score: {}".format(clusters_f1_scores[current_num_clusters]) )
        print( " - Adjusted Rand Score: {}".format(clusters_adjusted_rand_scores[current_num_clusters]) )
        
        print("\n")
        
    print("\n\n")