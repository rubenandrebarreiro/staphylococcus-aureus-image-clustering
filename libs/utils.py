# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:45:59 2020

@author: rubenandrebarreiro
"""

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import DataFrame Sub-Module,
# From Pandas Python's Library,
# as pandas_data_frame
from pandas import DataFrame as pandas_data_frame


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