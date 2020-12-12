# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:16:58 2020

@author: rubenandrebarreiro
"""

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library,
# as k_means
from sklearn.cluster import KMeans as k_means


from libs.visualization_and_plotting import plot_clusters_centroids_and_radii

from libs.visualization_and_plotting import plot_silhouette_analysis

from libs.visualization_and_plotting import plot_confusion_matrix_rand_index_clustering_heatmap

from libs.visualization_and_plotting import plot_clustering_scores


from libs.performance_scoring_metrics import compute_clustering_performance_metrics

from libs.performance_scoring_metrics import print_k_means_clustering_performance_metrics


def k_means_clustering_method(xs_features_data, num_clusters):
    
     k_means_clustering = k_means(n_clusters = num_clusters, init = 'k-means++', n_init = 10, max_iter = 300)
     
     k_means_clustering.fit(xs_features_data)
     
     ys_labels_predicted = k_means_clustering.predict(xs_features_data)
     
     clusters_centroids = k_means_clustering.cluster_centers_
     
     cluster_squared_error_sum_intertia = k_means_clustering.inertia_
     
     
     return ys_labels_predicted, clusters_centroids, cluster_squared_error_sum_intertia
     

def k_means_pre_clustering_method(xs_features_data, ys_labels_true, num_total_clusters):

    clusters_squared_errors_sums_intertias = matrix_array_zeros( num_total_clusters )
    
    clusters_silhouette_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_precision_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_recall_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_rand_index_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_f1_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_adjusted_rand_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    
    
    for current_num_clusters in range( 1, ( num_total_clusters + 1 ) ):
        
        ys_labels_predicted, clusters_centroids, error_k_means_pre_clustering = k_means_clustering_method(xs_features_data, current_num_clusters)
                
        clusters_squared_errors_sums_intertias[ ( current_num_clusters - 1 ) ] = error_k_means_pre_clustering
        
        plot_clusters_centroids_and_radii("K-Means", xs_features_data, ys_labels_predicted, clusters_centroids, num_clusters = current_num_clusters, epsilon = None, final_clustering = False)
        
        
        if(current_num_clusters >= 2):
            
            plot_silhouette_analysis("K-Means", xs_features_data, ys_labels_predicted, clusters_centroids, current_num_clusters, epsilon = None, final_clustering = False)
        
            silhouette_score, precision_score, recall_score, rand_index_score, f1_score, adjusted_rand_score, confusion_matrix_rand_index_clustering = compute_clustering_performance_metrics("K-Means", xs_features_data, ys_labels_true, ys_labels_predicted, current_num_clusters, final_clustering = False)
            
            plot_confusion_matrix_rand_index_clustering_heatmap("K-Means", confusion_matrix_rand_index_clustering, current_num_clusters, epsilon = None, final_clustering = False)
                        
            clusters_silhouette_scores[( current_num_clusters - 2 )] = silhouette_score
            clusters_precision_scores[( current_num_clusters - 2 )] = precision_score
            clusters_recall_scores[( current_num_clusters - 2 )] = recall_score
            clusters_rand_index_scores[( current_num_clusters - 2 )] = rand_index_score
            clusters_f1_scores[( current_num_clusters - 2 )] = f1_score
            clusters_adjusted_rand_scores[( current_num_clusters - 2 )] = adjusted_rand_score
    
    
    print_k_means_clustering_performance_metrics("K-Means", num_total_clusters, clusters_squared_errors_sums_intertias, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    plot_clustering_scores("K-Means", num_total_clusters, 0, 0, 0, None, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    
    return clusters_squared_errors_sums_intertias, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores



def k_means_final_clustering(xs_features_data, ys_labels_true, num_clusters = 5):
    
    ys_labels_predicted, k_means_estimator_centroids, k_means_final_clustering_error = k_means_clustering_method(xs_features_data, num_clusters)
    
    plot_clusters_centroids_and_radii("K-Means", xs_features_data, ys_labels_predicted, k_means_estimator_centroids, num_clusters = num_clusters, epsilon = None, final_clustering = True)
    
    
    if(num_clusters >= 2):
            
            plot_silhouette_analysis("K-Means", xs_features_data, ys_labels_predicted, k_means_estimator_centroids, num_clusters, epsilon = None, final_clustering = True)
            
            k_means_final_clustering_silhouette_score, k_means_final_clustering_precision_score, k_means_final_clustering_recall_score, k_means_final_clustering_rand_index_score, k_means_final_clustering_f1_score, k_means_final_clustering_adjusted_rand_score, k_means_final_clustering_confusion_matrix_rand_index = compute_clustering_performance_metrics("K-Means", xs_features_data, ys_labels_true, ys_labels_predicted, num_clusters, final_clustering = True)
            
            plot_confusion_matrix_rand_index_clustering_heatmap("K-Means", k_means_final_clustering_confusion_matrix_rand_index, num_clusters, epsilon = None, final_clustering = True)
            
    
    return k_means_final_clustering_error, k_means_final_clustering_silhouette_score, k_means_final_clustering_precision_score, k_means_final_clustering_recall_score, k_means_final_clustering_rand_index_score, k_means_final_clustering_f1_score, k_means_final_clustering_adjusted_rand_score, k_means_final_clustering_confusion_matrix_rand_index