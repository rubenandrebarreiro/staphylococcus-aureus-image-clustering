"""
@author: Martim Figueiredo - 42648
@author: Rúben André Barreiro - 42648
"""

# -------------------------------------------------------------------#


import warnings
warnings.filterwarnings("ignore")

from libs.install_libraries import install_required_system_libraries    

install_required_system_libraries("kneed")


# A) Libraries Used:

# - 1) General Libraries:

from libs.utils import create_data_frames_extraction

from libs.utils import compute_distances_nearest_neighbors


from libs.visualization_and_plotting import intialize_plotting_style

from libs.visualization_and_plotting import generate_data_analysis_plots

from libs.visualization_and_plotting import plot_elbow_method

from libs.visualization_and_plotting import plot_k_distance_method


from k_means_clustering import k_means_pre_clustering_method

from k_means_clustering import k_means_final_clustering


from dbscan_clustering import dbscan_pre_clustering_method

#from dbscan_clustering import dbscan_final_clustering_method


from kneed import KneeLocator as knee_locator


# Import loadtxt,
# From the NumPy's Python Library,
# as load_txt
from numpy import loadtxt as load_txt

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import Pre-Processing Sub-Module,
# From SciKit Learn's Python's Library,
# as normalize_data
from sklearn.preprocessing import normalize as normalize_data

# Import preprocessing.MinMaxScaler Sub-Module,
# from SciKit-Learn Python's Library,
# as min_max_scaler
from sklearn.preprocessing import MinMaxScaler as min_max_scaler

# - 2) Customised Libraries:

# Import images_as_matrix,
# From the TP2_Aux Customised Python Library,
# as images_as_matrix
from libs.tp2_aux import images_as_matrix as images_as_numpy_matrix

# Import report_clusters,
# From the TP2_Aux Customised Python Library,
# as html_report_cluster_labels
from libs.tp2_aux import report_clusters as html_report_cluster_labels


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


# - 4) Features' Selection Libraries:

# Import F Classifier,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as f_value_features
from sklearn.feature_selection import f_classif as f_value_features

# Import Select K Best,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as select_k_best_features
from sklearn.feature_selection import SelectKBest as select_k_best_features


# -------------------------------------------------------------------#

# A) CONSTANTS:
    
# The number of features/components to be extracted,
# from each Features' Extraction method
NUM_FEATURES_COMPONENTS = 6

# The threshold for the selection of the prior Best Features,
# from the F-Values, given by the F-Test (F1 Score)
F_VALUE_THRESHOLD = 10.0

# The maximum number of Clusters for the K-Means Clustering
NUM_MAX_CLUSTERS = 12

# The number of K Nearest Neighbors
NUM_K_NEAREST_NEIGHBORS = 5

# The start value of ε (Epsilon), for the DBScan Clustering
START_EPSILON = 0.01

# The end value of ε (Epsilon), for the DBScan Clustering
END_EPSILON = 0.285

# The step value of ε (Epsilon), for the DBScan Clustering
STEP_EPSILON = 0.01

# -------------------------------------------------------------------#



# B) PROGRAM:

# Step 1:
# - Initialise the global variables and prepare the data;


# The 2D NumPy Matrix, representing all the samples images,
# with an image per row (563 lines),
# and one indiviual pixel by column
# (50 pixels x 50 pixels = 2500 columns)
xs_images_matrix = images_as_numpy_matrix()
# The Total Number of Images/Examples (Samples)
num_total_images_examples = len(xs_images_matrix)

# The Identifiers and Labels for
# the Examples and their Celular Phases of
# the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
ids_and_labels = load_txt("labels.txt", delimiter = ",")

# The Labels for the Celular Phases of the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
ys_labels_true = ids_and_labels[:,1]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
xs_ids = ids_and_labels[:,0]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies,
# which are not labelled (Label 0)
xs_ids_not_labelled_data = xs_ids[ys_labels_true == 0]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
# which are labelled (Labels 1, 2, 3 for Celular Phases 1, 2, 3, respectively)
xs_ids_labelled_data = xs_ids[ys_labels_true != 0]


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


# Create Data Frames for the 3 Features Extractions (PCA, TSNE and Isomap)
data_frame_transformed_extraction_pca, data_frame_columns_pca, data_frame_transformed_extraction_tsne, data_frame_columns_tsne, data_frame_transformed_extraction_isomap, data_frame_columns_isomap = create_data_frames_extraction(transformed_xs_images_matrix_pca, transformed_xs_images_matrix_tsne, transformed_xs_images_matrix_isomap, ys_labels_true, num_total_images_examples = 563, num_features_components = NUM_FEATURES_COMPONENTS)

# Initialize the Visualization/Plotting Style
intialize_plotting_style('seaborn-dark')

# Generate Analysis' Plots, for several Visualization Plots
generate_data_analysis_plots(data_frame_transformed_extraction_pca, data_frame_columns_pca, data_frame_transformed_extraction_tsne, data_frame_columns_tsne, data_frame_transformed_extraction_isomap, data_frame_columns_isomap, num_components = NUM_FEATURES_COMPONENTS)


# The final Features Extracted, to be used, in the Clustering methods,
# filled initially with zeros (0s)
xs_features = matrix_array_zeros( (num_total_images_examples, ( 3 * NUM_FEATURES_COMPONENTS ) ) )
xs_features = matrix_array_zeros( (num_total_images_examples, ( ( 3 * NUM_FEATURES_COMPONENTS ) + 1 ) ) )

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


def f_value_anova(xs_features, ys_labels_true):
    
    f_values, probabilities_f_values = f_value_features(xs_features, ys_labels_true) 
    
    
    return f_values, probabilities_f_values



# Step 2:
# - Select the best features, based on a predefined threshold;
# Select the F-Values and the Probabilities, from the F-Test (F1 Score)
f_values, probabilities_f_values = f_value_anova(xs_features, ys_labels_true)


print("\n\n")
print("\n\n-----F Values----")
print(f_values)
print("\n\n")
print("-----F Values----\n\n")
print("\n\n-----F Probs----")
print(probabilities_f_values)
print("\n\n")
print("-----F probs----\n\n")


# The K Best Features, from the F-Values, given by the F-Test (F1 Score)
best_features_priori_indexes = []

# For all the indexes of the F-Values, given by the F-Test (F1 Score)
for current_feature_index in range( len(f_values) ):

    # If the current F-Value is higher than 10,
    # this feature will be considered
    if( f_values[current_feature_index] >= F_VALUE_THRESHOLD ):
        
        # Append the current index of the Feature
        best_features_priori_indexes.append(current_feature_index)


num_best_features_priori_indexes = len(best_features_priori_indexes)

xs_best_features_priori = matrix_array_zeros( ( num_total_images_examples, num_best_features_priori_indexes ) )

xs_best_features_priori_1 = matrix_array_zeros( ( num_total_images_examples, num_best_features_priori_indexes ) )
xs_best_features_priori_2 = matrix_array_zeros( ( num_total_images_examples, num_best_features_priori_indexes ) )

# For all the indexes of the K piori Best Features selected previously
for current_xs_best_features_priori_index in range(num_best_features_priori_indexes):

    # Select the K Best Features, from the initial Features extracted    
    xs_best_features_priori_1[:, current_xs_best_features_priori_index] = xs_features[:, best_features_priori_indexes[current_xs_best_features_priori_index]]
    

xs_best_features_priori_2 = select_k_best_features(f_value_features, k = num_best_features_priori_indexes).fit_transform(xs_features, ys_labels_true)


if(xs_best_features_priori_1.all() == xs_best_features_priori_2.all()):
    
    print("The Best K Features chosen manually and automatically, are the same, and will be used!!!\n\n")
    
    xs_best_features_priori = xs_best_features_priori_1


normalized_data_xs_best_features_priori = normalize_data(xs_best_features_priori)



# ---- K-Means Clustering ----

k_means_squared_errors_sums_intertias, k_means_silhouette_scores, k_means_precision_scores, k_means_recall_scores, k_means_rand_index_scores, k_means_f1_scores, k_means_adjusted_rand_scores = k_means_pre_clustering_method(normalized_data_xs_best_features_priori, ys_labels_true, num_total_clusters = NUM_MAX_CLUSTERS)

k_means_xs_points_elbow_method, k_means_ys_points_elbow_method = plot_elbow_method("K-Means", k_means_squared_errors_sums_intertias, num_max_clusters = NUM_MAX_CLUSTERS)

kneed_locator_elbow = knee_locator(k_means_xs_points_elbow_method, k_means_ys_points_elbow_method, S = 1.0, curve = "convex", direction = "decreasing")


final_num_clusters = round(kneed_locator_elbow.elbow, 0)

print( "The best K (Number of Clusters), for K-Means Clustering, found:" )
print( "- {}\n\n".format(final_num_clusters) )

error_k_means_final_clustering = k_means_final_clustering(normalized_data_xs_best_features_priori, ys_labels_true, num_clusters = final_num_clusters)

# ---- K-Means Clustering ----


# ---- DBScan Clustering ----

# TODO - faz diferença normalizar de novo os dados?!
min_max_scaler_data = min_max_scaler()
    
normalized_data_transformed_xs_best_features_priori = min_max_scaler_data.fit_transform(normalized_data_xs_best_features_priori)

k_neighbors_distances = compute_distances_nearest_neighbors(normalized_data_transformed_xs_best_features_priori, num_closest_k_neighbors = NUM_K_NEAREST_NEIGHBORS)

k_neighbors_distances = plot_k_distance_method("DBScan", k_neighbors_distances)

dbscan_num_centroids, dbscan_num_inliers, dbscan_num_outliers, dbscan_silhouette_scores, dbscan_precision_scores, dbscan_recall_scores, dbscan_rand_index_scores, dbscan_f1_scores, dbscan_adjusted_rand_scores = dbscan_pre_clustering_method(normalized_data_transformed_xs_best_features_priori, ys_labels_true, start_epsilon = START_EPSILON, end_epsilon = END_EPSILON, step_epsilon = STEP_EPSILON)

# ---- DBScan Clustering ----
