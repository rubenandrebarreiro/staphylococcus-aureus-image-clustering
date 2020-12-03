# -*- coding: utf-8 -*-
"""

@author: Martim Figueiredo - 42648
@author: Rúben André Barreiro - 42648

"""

# -------------------------------------------------------------------#


# A) Libraries Used:

# - 1) General Libraries:

# Import loadtxt,
# From the NumPy's Python Library,
# as load_txt
from numpy import loadtxt as load_txt

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros


# - 2) Customised Libraries:

# Import images_as_matrix,
# From the TP2_Aux Customised Python Library,
# as images_as_matrix
from tp2_aux import images_as_matrix as images_as_numpy_matrix

# Import report_clusters,
# From the TP2_Aux Customised Python Library,
# as html_report_cluster_labels
from tp2_aux import report_clusters as html_report_cluster_labels


# - 3) Libraries for Features Extraction:

# Import PCA,
# From the Decomposition Module,
# of the SciKit-Learn's Python Library,
# as pca_decomposition
from sklearn.decomposition import PCA as pca_decomposition

# Import TSNE,
# From the Manifold Module,
# of the SciKit-Learn's Python Library,
# as t_distributed_stochastic_neighbor_embedding
from sklearn.manifold import TSNE as t_distributed_stochastic_neighbor_embedding 

# Import Isomap,
# From the Manifold Module,
# of the SciKit-Learn's Python Library,
# as isometric_mapping
from sklearn.manifold import Isomap as isometric_mapping 


# - 4) Libraries for Scoring Performance:

# Import Function Classifier,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as f_1_score
from sklearn.feature_selection import f_classif as f_1_score


# - 5) Clustering Methods' Libraries:

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library,
# as k_means
from sklearn.cluster import KMeans as k_means

# Import cluster.DBSCAN Sub-Module,
# from SciKit-Learn Python's Library,
# as dbscan
from sklearn.cluster import DBSCAN as dbscan



# - 6) Scoring/Metrics' Libraries:

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library,
# as k_means
from sklearn.metrics import adjusted_rand_score as skl_adjusted_rand_score

# Import cluster.DBSCAN Sub-Module,
# from SciKit-Learn Python's Library,
# as dbscan
from sklearn.metrics import silhouette_score as skl_silhouette_score



# -------------------------------------------------------------------#


# B) CONSTANTS:
    
NUM_FEATURES_COMPONENTS = 6


# -------------------------------------------------------------------#


# C) PROGRAM:

    
# Step 1:
# - Initialise the global variables and prepare the data;

  
# The 2D NumPy Matrix, representing all the samples images,
# with an image per row (563 lines),
# and one indiviual pixel by column
# (50 pixels x 50 pixels = 2500 columns)
xs_images_matrix = images_as_numpy_matrix()


# The Total Number of Images/Examples
num_total_images_examples = len(xs_images_matrix)


# The Identifiers and Labels for
# the Examples and their Celular Phases of
# the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
ids_and_labels = load_txt("labels.txt", delimiter=",")

# The Labels for the Celular Phases of the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
ys_labels = ids_and_labels[:,1]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
xs_ids = ids_and_labels[:,0]


# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies,
# which are not labelled (Label 0)
xs_ids_not_labelled_data = xs_ids[ys_labels == 0]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
# which are labelled (Labels 1, 2, 3 for Celular Phases 1, 2, 3, respectively)
xs_ids_labelled_data = xs_ids[ys_labels != 0]



# Step 2:
# - Process of Features' Extraction;


# For each method of Features' Extraction
# (3 methods of Extraction of Features),
# extract 6 Features (a total of 3 x 6 = 18 features):

# - 1) PCA (Principal Component Analysis) Decomposition:
pca = pca_decomposition(n_components = NUM_FEATURES_COMPONENTS)

# - 2) TSNE (T-Distributed Stochastic Neighbor Embedding):
tsne = t_distributed_stochastic_neighbor_embedding(n_components = NUM_FEATURES_COMPONENTS, method = "exact")

# - 3) Isomap (Isometric Mapping):
isomap = isometric_mapping(n_components = NUM_FEATURES_COMPONENTS)


# Fit the PCA (Principal Component Analysis) Decomposition,
# with the 2D NumPy Matrix, representing all the images,
# and transform the respective data, for the Feature Extraction
pca.fit(xs_images_matrix)
transformed_xs_images_matrix_pca = pca.transform(xs_images_matrix)

# Fit the TSNE (T-Distributed Stochastic Neighbor Embedding),
# with the 2D NumPy Matrix, representing all the images,
# and transform the respective data, for the Feature Extraction
transformed_xs_images_matrix_tsne = tsne.fit_transform(xs_images_matrix)

# Fit the Isomap (Isometric Mapping),
# with the 2D NumPy Matrix, representing all the images,
# and transform the respective data, for the Feature Extraction
transformed_xs_images_matrix_isomap = isomap.fit_transform(xs_images_matrix)


# The final Features Extracted, to be used, in the Clustering methods,
# filled initially with zeros (0s)
xs_features = matrix_array_zeros( (num_total_images_examples, ( 3 * NUM_FEATURES_COMPONENTS ) ) )

# The final Features Extracted, to be used, in the Clustering methods,
# filled with the fitted and transformed data,
# from the PCA (Principal Component Analysis) Decomposition
xs_features[:, 0 : NUM_FEATURES_COMPONENTS] = transformed_xs_images_matrix_pca

# The final Features Extracted, to be used, in the Clustering methods,
# filled with the fitted and transformed data,
# from the TSNE (T-Distributed Stochastic Neighbor Embedding)
xs_features[:, NUM_FEATURES_COMPONENTS : ( 2 * NUM_FEATURES_COMPONENTS ) ] = transformed_xs_images_matrix_tsne

# The final Features Extracted, to be used, in the Clustering methods,
# filled with the fitted and transformed data,
# from the Isomap (Isometric Mapping)
xs_features[:, ( 2 * NUM_FEATURES_COMPONENTS ) : ( 3 * NUM_FEATURES_COMPONENTS ) ] = transformed_xs_images_matrix_isomap



# Step 2:
# - Process of Clustering;


# k clusters
k = 10

# e distancia de vizinhanca
e = 10

for num_clusters in range(k):

    k_means_clustering = k_means(num_clusters)
            
    k_means_clustering.fit(xs_features)
    
    k_means_clustering_predicted_clusters_labels = k_means_clustering.predict(xs_images_matrix)
    
    centroids = k_means_clustering.cluster_centers_
    
    
    for neighborhood_distance in range(e):
    
    
        # Confirmar se é para usar todas as labels ou so labels diferentes de 0 (c/ fase atribuida)
        skl_adjusted_rand_score(ys_labels, k_means_clustering_predicted_clusters_labels)        
        
        skl_silhouette_score(xs_images_matrix)
        
        dbscan(algorithm=...)
        



# -------------------------------------------------------------------#