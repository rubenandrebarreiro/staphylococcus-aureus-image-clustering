# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:16:58 2020

@author: rubenandrebarreiro
"""

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import arange,
# From the NumPy's Python Library,
# as array_matrix
from numpy import array as array_matrix

# Import empty,
# From the NumPy's Python Library,
# as array_empty
from numpy import empty as array_empty

# Import diff,
# From the NumPy's Python Library,
# as array_diff
from numpy import diff as array_diff

# Import unique,
# From the NumPy's Python Library,
# as array_unique_values
from numpy import unique as array_unique_values

# Import mean,
# From the NumPy's Python Library,
# as array_mean
from numpy import mean as array_mean

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


# Import report_clusters,
# From the TP2_Aux Customised Python Library,
# as html_report_cluster_labels
from libs.tp2_aux import report_clusters as html_report_cluster_labels

# Import report_clusters,
# From the TP2_Aux Customised Python Library,
# as html_report_cluster_labels
from libs.tp2_aux import report_clusters_hierarchical as html_report_cluster_labels_hierarchical


def k_means_clustering_method(xs_features_data, num_clusters = 2, max_iterations = 300):
    
     k_means_clustering = k_means(n_clusters = num_clusters, init = 'k-means++', n_init = 10, max_iter = max_iterations)
     
     k_means_clustering.fit(xs_features_data)
     
     ys_labels_predicted = k_means_clustering.predict(xs_features_data)
     
     clusters_centroids = k_means_clustering.cluster_centers_
     
     cluster_squared_error_sum_intertia = k_means_clustering.inertia_
     
     
     return ys_labels_predicted, clusters_centroids, cluster_squared_error_sum_intertia
     

def k_means_pre_clustering(xs_features_data, ys_labels_true, num_total_clusters):

    clusters_squared_errors_sums_intertias = matrix_array_zeros( num_total_clusters )
    
    clusters_silhouette_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_precision_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_recall_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_rand_index_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_f1_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    clusters_adjusted_rand_scores = matrix_array_zeros( ( num_total_clusters - 1 ) )
    
    
    for current_num_clusters in range( 1, ( num_total_clusters + 1 ) ):
        
        ys_labels_predicted, clusters_centroids, error_k_means_pre_clustering = k_means_clustering_method(xs_features_data, num_clusters = current_num_clusters, max_iterations = 300)
                
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


def k_means_final_clustering(xs_features_data, ys_labels_true, num_clusters = 4):
    
    ys_labels_predicted, k_means_estimator_centroids, k_means_final_clustering_error = k_means_clustering_method(xs_features_data, num_clusters = num_clusters, max_iterations = 300)
    
    plot_clusters_centroids_and_radii("K-Means", xs_features_data, ys_labels_predicted, k_means_estimator_centroids, num_clusters = num_clusters, epsilon = None, final_clustering = True)
    
    
    if(num_clusters >= 2):
            
            plot_silhouette_analysis("K-Means", xs_features_data, ys_labels_predicted, k_means_estimator_centroids, num_clusters, epsilon = None, final_clustering = True)
            
            k_means_final_clustering_silhouette_score, k_means_final_clustering_precision_score, k_means_final_clustering_recall_score, k_means_final_clustering_rand_index_score, k_means_final_clustering_f1_score, k_means_final_clustering_adjusted_rand_score, k_means_final_clustering_confusion_matrix_rand_index = compute_clustering_performance_metrics("K-Means", xs_features_data, ys_labels_true, ys_labels_predicted, num_clusters, final_clustering = True)
            
            plot_confusion_matrix_rand_index_clustering_heatmap("K-Means", k_means_final_clustering_confusion_matrix_rand_index, num_clusters, epsilon = None, final_clustering = True)
            
    
    xs_ids_examples = list(range(0, len(xs_features_data)))
    
    html_report_cluster_labels(array_matrix(xs_ids_examples), ys_labels_predicted, "k-means.html")
    
    
    return k_means_final_clustering_error, k_means_final_clustering_silhouette_score, k_means_final_clustering_precision_score, k_means_final_clustering_recall_score, k_means_final_clustering_rand_index_score, k_means_final_clustering_f1_score, k_means_final_clustering_adjusted_rand_score, k_means_final_clustering_confusion_matrix_rand_index


def find_best_num_clusters(dbscan_xs_points_k_distance_method):
    
    dbscan_xs_points_variations_k_distance_method = abs(array_diff(dbscan_xs_points_k_distance_method))
    
    dbscan_xs_points_variations_mean_k_distance_method = array_mean(dbscan_xs_points_variations_k_distance_method)
    
    
    for current_num_clusters in range(len(dbscan_xs_points_variations_k_distance_method)):
        
        if(dbscan_xs_points_variations_k_distance_method[current_num_clusters] < dbscan_xs_points_variations_mean_k_distance_method):
    
            return ( current_num_clusters + 1 )
        
        
def bissect_k_means_into_two_sub_clusters(cluster_data_to_be_divided, cluster_examples_ids_to_be_divided, left_leaf_cluster_id_offset = 0, right_leaf_cluster_id_offset = 0):

    ys_labels_predicted, clusters_centroids, cluster_squared_error_sum_intertia = k_means_clustering_method(cluster_data_to_be_divided, num_clusters = 2, max_iterations = 300)
    
    
    two_sub_clusters_ids = array_unique_values(ys_labels_predicted)    
    two_sub_clusters_examples_ids = []
    two_sub_clusters_data = []
    
    ys_labels_predicted_without_offset = ys_labels_predicted
    ys_labels_predicted_with_offset = ys_labels_predicted
    
    
    ys_labels_predicted_with_offset[ys_labels_predicted_with_offset == 0] = ( ys_labels_predicted_with_offset[ys_labels_predicted_with_offset == 0] + left_leaf_cluster_id_offset )
    ys_labels_predicted_with_offset[ys_labels_predicted_with_offset == 1] = ( ys_labels_predicted_with_offset[ys_labels_predicted_with_offset == 1] + right_leaf_cluster_id_offset )
    
    
    for sub_cluster_id in range(2):
    
        if(sub_cluster_id == 0):
            
            two_sub_clusters_ids[sub_cluster_id] = ( two_sub_clusters_ids[sub_cluster_id] + left_leaf_cluster_id_offset )
        
        else:
            
            two_sub_clusters_ids[sub_cluster_id] = ( two_sub_clusters_ids[sub_cluster_id] + right_leaf_cluster_id_offset )
            
    
        two_sub_clusters_examples_ids.append(cluster_examples_ids_to_be_divided[ys_labels_predicted == two_sub_clusters_ids[sub_cluster_id]])
        two_sub_clusters_data.append(cluster_data_to_be_divided[ys_labels_predicted == two_sub_clusters_ids[sub_cluster_id]])
        
    
    return two_sub_clusters_ids, two_sub_clusters_examples_ids, two_sub_clusters_data, ys_labels_predicted_without_offset, ys_labels_predicted_with_offset, clusters_centroids, cluster_squared_error_sum_intertia

        
        
def bisecting_k_means_clustering(xs_features_data, examples_ids, final_num_clusters = 2, max_iterations = 100):
    
    clusters_data = [xs_features_data]
    clusters_examples_ids = [examples_ids]
    
    clusters_ids = [0]
    num_clusters = 1
    
    
    tree_predictions_lists = array_empty( (len(xs_features_data), 0) ).tolist()
    
    
    current_iteration = 0
    
    while ( ( num_clusters < final_num_clusters ) and ( current_iteration < max_iterations ) ):
       
        cluster_index_with_more_examples = -1
        num_max_examples_in_cluster = -1
        
        
        for index_cluster in range(num_clusters):
            
            num_examples_in_cluster = len(clusters_data[index_cluster])
            
                        
            if(num_examples_in_cluster > num_max_examples_in_cluster):
                
                cluster_index_with_more_examples = index_cluster
                num_max_examples_in_cluster = num_examples_in_cluster

        
        cluster_id_to_be_divided = clusters_ids[cluster_index_with_more_examples]
        cluster_data_to_be_divided = clusters_data[cluster_index_with_more_examples]
        cluster_examples_ids_to_be_divided = clusters_examples_ids[cluster_index_with_more_examples]
        
        clusters_ids.remove(cluster_id_to_be_divided)
        clusters_data.remove(cluster_data_to_be_divided)
        clusters_examples_ids.remove(cluster_examples_ids_to_be_divided)
        
        
        if(num_clusters == 1):

            two_sub_clusters_ids, two_sub_clusters_examples_ids, two_sub_clusters_data, ys_labels_predicted_without_offset, ys_labels_predicted_with_offset, clusters_centroids, cluster_squared_error_sum_intertia = bissect_k_means_into_two_sub_clusters(cluster_data_to_be_divided, cluster_examples_ids_to_be_divided, left_leaf_cluster_id_offset = 0, right_leaf_cluster_id_offset = 0)
        
        else:
            
            two_sub_clusters_ids, two_sub_clusters_examples_ids, two_sub_clusters_data, ys_labels_predicted_without_offset, ys_labels_predicted_with_offset, clusters_centroids, cluster_squared_error_sum_intertia = bissect_k_means_into_two_sub_clusters(cluster_data_to_be_divided, cluster_examples_ids_to_be_divided, left_leaf_cluster_id_offset = cluster_id_to_be_divided, right_leaf_cluster_id_offset = ( num_clusters - 1 ) )            
            
            
        
        for sub_cluster_id in range(2):
        
            clusters_ids.append(two_sub_clusters_ids[sub_cluster_id])

            clusters_data.append(two_sub_clusters_data[sub_cluster_id])

            clusters_examples_ids.append(two_sub_clusters_examples_ids[sub_cluster_id])
            
            
            for example_index in range(len(xs_features_data)):
                
                if(example_index in two_sub_clusters_examples_ids[sub_cluster_id]):
                    
                    tree_predictions_lists[example_index].append(sub_cluster_id)
        
        
        num_clusters = ( num_clusters + 1 )
        
        current_iteration = ( current_iteration + 1 )
        
    
        xs_ids_examples = list(range(0, len(examples_ids)))
        
        html_report_cluster_labels_hierarchical(xs_ids_examples, tree_predictions_lists, "bisecting-k-means-hierarchical.html")

        
    return clusters_ids, clusters_data, tree_predictions_lists