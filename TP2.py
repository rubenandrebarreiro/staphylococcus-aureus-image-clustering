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

# Import PyPlot Sub-Module,
# From Matplotlib Python's Library,
# as py_plot
from matplotlib import pyplot as py_plot

# Import Plotting Sub-Module,
# From Pandas Python's Library,
# as pandas_plot
from pandas import plotting as pandas_plot

# Import DataFrame Sub-Module,
# From Pandas Python's Library,
# as pandas_data_frame
from pandas import DataFrame as pandas_data_frame



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


# - 4) Features' Selection and Scoring Performances' Libraries:

# Import Function Classifier,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as f_1_score
from sklearn.feature_selection import f_classif as f_1_score


# Import Select K Best,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as select_k_best_features
from sklearn.feature_selection import SelectKBest as select_k_best_features


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

import warnings
warnings.filterwarnings("ignore")




# -------------------------------------------------------------------#


# B) CONSTANTS:
    
# The number of features/components to be extracted,
# from each Features' Extraction method
NUM_FEATURES_COMPONENTS = 6

# The threshold for the selection of the prior Best Features,
# from the F-Values, given by the F-Test (F1 Score)
f_value_threshold = 10.0


# -------------------------------------------------------------------#

# The Function to create the Data Frames for the Features' Extractions' Datasets 
def create_data_frames_extraction(transformed_xs_images_matrix_pca, transformed_xs_images_matrix_tsne, transformed_xs_images_matrix_isomap, ys_labels, num_total_images_examples = 563):
    
    # The Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the PCA (Principal Component Analysis) Features' Extraction method
    transformed_data_images_matrix_pca = matrix_array_zeros( (num_total_images_examples, ( NUM_FEATURES_COMPONENTS + 1 ) ) )
    
    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the PCA (Principal Component Analysis) Features' Extraction method, with the Features extracted
    transformed_data_images_matrix_pca[:, 0 : -1] = transformed_xs_images_matrix_pca
    
    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the PCA (Principal Component Analysis) Features' Extraction method, with the associated Labels
    transformed_data_images_matrix_pca[:, -1] = ys_labels
    
    # The Columns, with respect to the Identifiers of the Features and Labels,
    # to be used in the Data Frame, for the PCA (Principal Component Analysis) method
    data_frame_columns_pca = ['PCA - Feature 1', 'PCA - Feature 2', 'PCA - Feature 3',
                              'PCA - Feature 4', 'PCA - Feature 5', 'PCA - Feature 6',
                              'Celular Phase']
    
    # The Data Frame of the Features Extracted, for the PCA (Principal Component Analysis) method 
    data_frame_extraction_pca = pandas_data_frame(transformed_data_images_matrix_pca, columns = data_frame_columns_pca)   
    
    
    
    
    # The Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the TSNE (T-Distributed Stochastic Neighbor Embedding) Features' Extraction method
    transformed_data_images_matrix_tsne = matrix_array_zeros( (num_total_images_examples, ( NUM_FEATURES_COMPONENTS + 1 ) ) )
    
    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the TSNE (T-Distributed Stochastic Neighbor Embedding) Features' Extraction method, with the Features extracted
    transformed_data_images_matrix_tsne[:, 0 : -1] = transformed_xs_images_matrix_tsne
    
    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the TSNE (T-Distributed Stochastic Neighbor Embedding) Features' Extraction method, with the associated Labels
    transformed_data_images_matrix_tsne[:, -1] = ys_labels
    
    # The Columns, with respect to the Identifiers of the Features and Labels,
    # to be used in the Data Frame, for the TSNE (T-Distributed Stochastic Neighbor Embedding) method
    data_frame_columns_tsne = ['TSNE - Feature 7', 'TSNE - Feature 8', 'TSNE - Feature 9',
                               'TSNE - Feature 10', 'TSNE - Feature 11', 'TSNE - Feature 12',
                               'Celular Phase']    
    
    # The Data Frame of the Features Extracted, for the TSNE (T-Distributed Stochastic Neighbor Embedding) method 
    data_frame_extraction_tsne = pandas_data_frame(transformed_data_images_matrix_tsne, columns = data_frame_columns_tsne)   
    
    
    
    
    # The Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the Isomap (Isometric Mapping) Features' Extraction method
    transformed_data_images_matrix_isomap = matrix_array_zeros( (num_total_images_examples, ( NUM_FEATURES_COMPONENTS + 1 ) ) )
    
    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the Isomap (Isometric Mapping) Features' Extraction method, with the Features extracted
    transformed_data_images_matrix_isomap[:, 0 : -1] = transformed_xs_images_matrix_isomap
    
    # Fill the Matrix for the Features and Labels, to be used to create the Data Frame,
    # for the Isomap (Isometric Mapping) Features' Extraction method, with the associated Labels
    transformed_data_images_matrix_isomap[:, -1] = ys_labels
    
    # The Columns, with respect to the Identifiers of the Features and Labels,
    # to be used in the Data Frame, for the Isomap (Isometric Mapping) method
    data_frame_columns_isomap = ['Isomap - Feature 13', 'Isomap - Feature 14', 'Isomap - Feature 15',
                                 'Isomap - Feature 16', 'Isomap - Feature 17', 'Isomap - Feature 18',
                                 'Celular Phase']    
    
    # The Data Frame of the Features Extracted, for the Isomap (Isometric Mapping) method 
    data_frame_extraction_isomap = pandas_data_frame(transformed_data_images_matrix_isomap, columns = data_frame_columns_isomap)   
    
    
    
    return data_frame_extraction_pca, data_frame_columns_pca, data_frame_extraction_tsne, data_frame_columns_tsne, data_frame_extraction_isomap, data_frame_columns_isomap


# The Function to plot the Stacked Histograms
def plot_stacked_histograms(data_frame_transformed_extraction, method, num_components = 6, alpha_value = 0.8):
    
    # Initialise the Plot
    py_plot.figure( figsize = (15, 12), frameon = True )

    # Plot the Stacked Histograms
    data_frame_transformed_extraction.plot( kind = 'hist', bins = 15, alpha = alpha_value )
        
    # Set the label for the X axis of the Plot
    py_plot.xlabel("Individual Features")
    
    # Set the label for the Y axis of the Plot
    py_plot.ylabel("Frequency")
    
    # Set the Title of the Plot
    py_plot.title( 'Frequency of Individual Features,\nfor {} with {} Components, with alpha={}'.format(method, num_components, alpha_value) )
    
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/{}-with-{}-components-data-stacked-histograms-alpha-{}.png'.format(method.lower(), num_components, alpha_value), dpi = 600, bbox_inches = 'tight' )
    
    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()
    
    # Show the Plot
    py_plot.show()
    
    # Close the Plot
    py_plot.close()


# The Function to plot the Individual Histograms
def plot_individual_histograms(data_frame_transformed_extraction, method, num_components = 6, alpha_value = 0.8):
    
    # Initialise the Plot
    py_plot.figure( figsize = (15, 12), frameon = True )
    
    # Plot the Individual Histograms
    data_frame_transformed_extraction.hist( bins = 15, alpha = alpha_value, layout = (2, 3) )
    
    # Set the aspect of the Plot
    py_plot.gca().set_aspect( 'auto', adjustable = 'box' )
    
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/{}-with-{}-components-data-individual-histograms-alpha-{}.png'.format(method.lower(), num_components, alpha_value), dpi = 600, bbox_inches = 'tight' )
    
    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()
    
    # Show the Plot
    py_plot.show()
    
    # Close the Plot
    py_plot.close()
    
    
# The Function to plot the Box
def plot_box(data_frame_transformed_extraction, method, num_components = 6):
    
    # Initialise the Plot
    py_plot.figure( figsize = (15, 15), frameon = True, dpi = 600)

    # Plot the Box
    data_frame_transformed_extraction.plot( kind = 'box', rot = 20)
    
    # Set the Title of the Plot
    py_plot.title( 'Box Plot, for {} with {} Components'.format(method, num_components) )
    
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/{}-with-{}-components-data-box.png'.format(method.lower(), num_components), dpi = 600, bbox_inches = 'tight' )
    
    # Autoscale the Layout of the Plot to tight
    py_plot.autoscale()
    
    # Show the Plot
    py_plot.show()
    
    # Close the Plot
    py_plot.close()
    
    
# The Function to plot the Scatter Matrix, with a given type Diagonal (KDE or Histogram)
def plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction, method, diagonal_plot, num_components = 6, alpha_value = 0.8):
    
    # Initialise the Plot
    py_plot.figure( figsize = (15, 12), frameon = True )

    # Plot the Scatter Matrix, with a given type Diagonal (KDE or Histogram)
    pandas_plot.scatter_matrix( data_frame_transformed_extraction, alpha = alpha_value, figsize = (15,10), diagonal = diagonal_plot )
    
    # Save the Plot, as a figure/image
    py_plot.savefig('imgs/plots/{}-with-{}-components-data-scatter-matrix-{}-diagonal-alpha-{}.png'.format(method.lower(), num_components, diagonal_plot, alpha_value), dpi = 600, bbox_inches = 'tight' )
    
    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()
    
    # Show the Plot
    py_plot.show()
    
    # Close the Plot
    py_plot.close()
    
    

# The Function to plot the Parallel Coordinates
def plot_parallel_coordinates(data_frame_transformed_extraction, method, num_components = 6):
    
    # Initialise the Plot
    py_plot.figure( figsize = (15, 12), frameon = True )

    # Plot the Parallel Coordinates
    pandas_plot.parallel_coordinates(data_frame_transformed_extraction, class_column = 'Celular Phase')
    
    # Set the Title of the Plot
    py_plot.title( 'Parallel Coordinates, for {} with {} Components,\nrepresenting the different Classes in different Colours'.format(method, num_components) )
    
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/{}-with-{}-components-data-parallel-coordinates.png'.format(method.lower(), num_components), dpi = 600, bbox_inches = 'tight' )
    
    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()
    
    # Show the Plot
    py_plot.show()
    
    # Close the Plot
    py_plot.close()
    

# The Function to plot the Andrew's Curves
def plot_andrews_curves(data_frame_transformed_extraction, method, num_components = 6):
    
    # Initialise the Plot
    py_plot.figure( figsize = (15, 12), frameon = True )

    # Plot the Parallel Coordinates
    pandas_plot.andrews_curves(data_frame_transformed_extraction, class_column = 'Celular Phase')
    
    # Set the Title of the Plot
    py_plot.title( 'Andrew\'s Curves, for {} with {} Components,\nrepresenting the different Classes in different Colours'.format(method, num_components) )
    
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/{}-with-{}-components-data-andrew-curves.png'.format(method.lower(), num_components), dpi = 600, bbox_inches = 'tight' )
    
    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()
    
    # Show the Plot
    py_plot.show()
    
    # Close the Plot
    py_plot.close()










# C) PROGRAM:

    
# Step 1:
# - Initialise the global variables and prepare the data;


# Set the Style of the Plots, as 'Seaborn Dark' Style
py_plot.style.use('seaborn-dark')

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




print("\n\n-----Image matrix Features----")
print(xs_images_matrix)
print("-----Image matrix Features----\n\n")



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
data_frame_transformed_extraction_pca, data_frame_columns_pca, data_frame_transformed_extraction_tsne, data_frame_columns_tsne, data_frame_transformed_extraction_isomap, data_frame_columns_isomap = create_data_frames_extraction(transformed_xs_images_matrix_pca, transformed_xs_images_matrix_tsne, transformed_xs_images_matrix_isomap, ys_labels, num_total_images_examples = 563)


plot_stacked_histograms(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_stacked_histograms(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_stacked_histograms(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)

plot_individual_histograms(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_individual_histograms(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_individual_histograms(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)

plot_box(data_frame_transformed_extraction_pca[data_frame_columns_pca[0:-1]], "PCA", num_components = NUM_FEATURES_COMPONENTS)
plot_box(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", num_components = NUM_FEATURES_COMPONENTS)
plot_box(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", num_components = NUM_FEATURES_COMPONENTS)

plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", "kde", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", "kde", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", "kde", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)

plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", "hist", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", "hist", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)
plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", "hist", num_components = NUM_FEATURES_COMPONENTS, alpha_value = 0.8)

plot_parallel_coordinates(data_frame_transformed_extraction_pca, "PCA", num_components = 6)
plot_parallel_coordinates(data_frame_transformed_extraction_tsne, "TSNE", num_components = 6)
plot_parallel_coordinates(data_frame_transformed_extraction_isomap, "Isomap", num_components = 6)

plot_andrews_curves(data_frame_transformed_extraction_pca, "PCA", num_components = 6)
plot_andrews_curves(data_frame_transformed_extraction_tsne, "TSNE", num_components = 6)
plot_andrews_curves(data_frame_transformed_extraction_isomap, "Isomap", num_components = 6)


# The final Features Extracted, to be used, in the Clustering methods,
# filled initially with zeros (0s)
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


# Step 2:
# - Select the best features, based on a predefined threshold;

# Select the F-Values and the Probabilities, from the F-Test (F1 Score)
f_values, probabilities_f_values = f_1_score(xs_features, ys_labels)


print("\n\n-----F Values----")
print(f_values)
print("-----F Values----\n\n")
print("\n\n-----F Probs----")
print(probabilities_f_values)
print("-----F probs----\n\n")


# The K Best Features, from the F-Values, given by the F-Test (F1 Score)
best_features_priori_indexes = []

# For all the indexes of the F-Values, given by the F-Test (F1 Score)
for current_feature_index in range( len(f_values) ):

    # If the current F-Value is higher than 10,
    # this feature will be considered
    if( f_values[current_feature_index] >= f_value_threshold ):
        
        # Append the current index of the Feature
        best_features_priori_indexes.append(current_feature_index)


print(best_features_priori_indexes)

num_best_features_priori_indexes = len(best_features_priori_indexes)

xs_best_features_priori_1 = matrix_array_zeros( ( num_total_images_examples, num_best_features_priori_indexes ) )

print("\n\n-----18 Features----")
print(xs_features)
print("-----18 Features----\n\n")


# For all the indexes of the K piori Best Features selected previously
for current_xs_best_features_priori_index in range(num_best_features_priori_indexes):
    
    # Select the K Best Features, from the initial Features extracted    
    xs_best_features_priori_1[:, current_xs_best_features_priori_index] = xs_features[:, best_features_priori_indexes[current_xs_best_features_priori_index]]
    

xs_best_features_priori_2 = select_k_best_features(f_1_score, k = num_best_features_priori_indexes).fit_transform(xs_features, ys_labels)


print("\n\n-----K Best Features----")
print(xs_best_features_priori_1)
print("-----K Best Features----\n\n")

    
if(xs_best_features_priori_1.all() == xs_best_features_priori_2.all()):
    print("The Best K Features chosen manually and automatically, are the same, and will be used!!!")



"""
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
        

"""

# -------------------------------------------------------------------#