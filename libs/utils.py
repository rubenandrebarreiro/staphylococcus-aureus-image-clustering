# -*- coding: utf-8 -*-
"""

Last update on Sun Nov 20 20:00:00 2020

@student-name: Martim Cevadinha Figueiredo
@student-email: mc.figueiredo@campus.fct.unl.pt
@student-number: 52701

@student-name: Ruben Andre Barreiro
@student-email: r.barreiro@campus.fct.unl.pt
@student-number: 42648

@degree: Master of Computer Science and Engineering (MIEI)

@college: NOVA School of Science and Technology (FCT NOVA)
@university: New University of Lisbon (UNL)

"""

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import sort,
# From the NumPy's Python Library,
# as ordering_sort
from numpy import sort as ordering_sort

# Import mean,
# From the NumPy's Python Library,
# as array_mean
from numpy import mean as array_mean

# Import std,
# From the NumPy's Python Library,
# as array_std
from numpy import std as array_std

# Import max,
# From the NumPy's Python Library,
# as array_matrix_max
from numpy import max as array_matrix_max

# Import min,
# From the NumPy's Python Library,
# as array_matrix_min
from numpy import min as array_matrix_min

# Import DataFrame Sub-Module,
# From Pandas Python's Library,
# as pandas_data_frame
from pandas import DataFrame as pandas_data_frame

# Import F Classifier,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as f_value_features
from sklearn.feature_selection import f_classif as f_value_features

# Import Neighbors.NearestNeighbors Sub-Module,
# from SciKit-Learn Python's Library,
# as nearest_neighbors
from sklearn.neighbors import NearestNeighbors as skl_nearest_neighbors


def standartize_data(xs_data_points):
    
    means = array_mean(xs_data_points, axis = 0)
    stdevs = array_std(xs_data_points, axis = 0)
    
    xs_data_standartized = ( ( xs_data_points - means ) / stdevs )

    return xs_data_standartized

def normalize_data(xs_data_points):
        
    xs_data_points_max = array_matrix_max(xs_data_points, axis = 0)
    xs_data_points_min = array_matrix_min(xs_data_points, axis = 0)

    xs_data_points_normalized = ( ( xs_data_points - xs_data_points_min ) / ( xs_data_points_max - xs_data_points_min ) )

    
    return xs_data_points_normalized


def f_value_anova(xs_features, ys_labels_true):
    
    f_test_values, f_test_probabilities = f_value_features(xs_features, ys_labels_true) 
    
    
    return f_test_values, f_test_probabilities


def compute_arg_majority(xs_data, threshold_majority):
    
    for index in range(len(xs_data)):
        
        if(xs_data[index] > threshold_majority):
            
            return index
        

def compute_distances_nearest_neighbors(xs_features_data, num_closest_k_neighbors = 5):
    
    nearest_neighbors = skl_nearest_neighbors(n_neighbors = num_closest_k_neighbors)
    
    neighbors = nearest_neighbors.fit(xs_features_data)
    
    
    k_neighbors_distances, k_neighbors_indices = neighbors.kneighbors(xs_features_data)
    
    k_neighbors_distances = ordering_sort(k_neighbors_distances, axis = 0)
    
    k_neighbors_distances_epsilons = k_neighbors_distances[:, 1]
    
    
    num_data_points_sorted_by_distance = len(k_neighbors_distances_epsilons)
    
    
    return num_data_points_sorted_by_distance, k_neighbors_distances_epsilons


# The Function to create the Data Frames for the Features' Extractions' Datasets 
def create_data_frames_extraction(transformed_xs_images_matrix_pca, transformed_xs_images_matrix_tsne, transformed_xs_images_matrix_isomap, ys_labels_true, num_total_images_examples = 563, num_features_components = 6):

    # The Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the PCA (Principal Component Analysis) Features' Extraction method
    transformed_data_images_matrix_pca = matrix_array_zeros( (num_total_images_examples, ( num_features_components + 1 ) ) )

    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the PCA (Principal Component Analysis) Features' Extraction method, with the Features extracted
    transformed_data_images_matrix_pca[:, 0 : -1] = transformed_xs_images_matrix_pca

    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the PCA (Principal Component Analysis) Features' Extraction method, with the associated Labels
    transformed_data_images_matrix_pca[:, -1] = ys_labels_true

    # The Columns, with respect to the Identifiers of the Features and Labels,
    # to be used in the Data Frame, for the PCA (Principal Component Analysis) method
    data_frame_columns_pca = ['PCA - Feature 1', 'PCA - Feature 2', 'PCA - Feature 3',
                              'PCA - Feature 4', 'PCA - Feature 5', 'PCA - Feature 6',
                              'Celular Phase']

    # The Data Frame of the Features Extracted, for the PCA (Principal Component Analysis) method 
    data_frame_extraction_pca = pandas_data_frame(transformed_data_images_matrix_pca, columns = data_frame_columns_pca)   




    # The Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the TSNE (T-Distributed Stochastic Neighbor Embedding) Features' Extraction method
    transformed_data_images_matrix_tsne = matrix_array_zeros( (num_total_images_examples, ( num_features_components + 1 ) ) )

    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the TSNE (T-Distributed Stochastic Neighbor Embedding) Features' Extraction method, with the Features extracted
    transformed_data_images_matrix_tsne[:, 0 : -1] = transformed_xs_images_matrix_tsne

    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the TSNE (T-Distributed Stochastic Neighbor Embedding) Features' Extraction method, with the associated Labels
    transformed_data_images_matrix_tsne[:, -1] = ys_labels_true

    # The Columns, with respect to the Identifiers of the Features and Labels,
    # to be used in the Data Frame, for the TSNE (T-Distributed Stochastic Neighbor Embedding) method
    data_frame_columns_tsne = ['TSNE - Feature 7', 'TSNE - Feature 8', 'TSNE - Feature 9',
                               'TSNE - Feature 10', 'TSNE - Feature 11', 'TSNE - Feature 12',
                               'Celular Phase']    

    # The Data Frame of the Features Extracted, for the TSNE (T-Distributed Stochastic Neighbor Embedding) method 
    data_frame_extraction_tsne = pandas_data_frame(transformed_data_images_matrix_tsne, columns = data_frame_columns_tsne)   




    # The Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the Isomap (Isometric Mapping) Features' Extraction method
    transformed_data_images_matrix_isomap = matrix_array_zeros( (num_total_images_examples, ( num_features_components + 1 ) ) )

    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the Isomap (Isometric Mapping) Features' Extraction method, with the Features extracted
    transformed_data_images_matrix_isomap[:, 0 : -1] = transformed_xs_images_matrix_isomap

    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the Isomap (Isometric Mapping) Features' Extraction method, with the associated Labels
    transformed_data_images_matrix_isomap[:, -1] = ys_labels_true

    # The Columns, with respect to the Identifiers of the Features and Labels,
    # to be used in the Data Frame, for the Isomap (Isometric Mapping) method
    data_frame_columns_isomap = ['Isomap - Feature 13', 'Isomap - Feature 14', 'Isomap - Feature 15',
                                 'Isomap - Feature 16', 'Isomap - Feature 17', 'Isomap - Feature 18',
                                 'Celular Phase']    

    # The Data Frame of the Features Extracted, for the Isomap (Isometric Mapping) method 
    data_frame_extraction_isomap = pandas_data_frame(transformed_data_images_matrix_isomap, columns = data_frame_columns_isomap)   



    return data_frame_extraction_pca, data_frame_columns_pca, data_frame_extraction_tsne, data_frame_columns_tsne, data_frame_extraction_isomap, data_frame_columns_isomap