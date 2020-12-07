"""
@author: Martim Figueiredo - 42648
@author: Rúben André Barreiro - 42648
"""

# -------------------------------------------------------------------#

# Import PyPlot Sub-Module,
# From Matplotlib Python's Library,
# as py_plot
from matplotlib import pyplot as py_plot

# Import Patches Sub-Module,
# From Matplotlib Python's Library,
# as matplotlib_patches
from matplotlib import patches as matplotlib_patches

# Import Subtract Sub-Module,
# From NumPy Python's Library,
# as subtract_nums
from numpy import subtract as subtract_numbers

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros


# Import LinearAlgebra.Norm Sub-Module,
# From NumPy Python's Library,
# as norm_number
from numpy.linalg import norm as norm_number

# Import Plotting Sub-Module,
# From Pandas Python's Library,
# as pandas_plot
from pandas import plotting as pandas_plot


COLORS_MATPLOTLIB = ['red', 'darkorange', 'goldenrod', 'yellow', 'lawngreen', 'forestgreen', 'turquoise', 'teal', 'deepskyblue', 'midnightblue', 'blue', 'darkviolet', 'magenta', 'pink']


# The Function to initialize the Visualization/Plotting Style
def intialize_plotting_style(plotting_style = 'seaborn-dark'):

    # Set the Style of the Plots, as 'Seaborn Dark' Style
    py_plot.style.use(plotting_style)


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
    py_plot.savefig( 'imgs/plots/data-analysis-visualization/{}-with-{}-components-data-stacked-histograms-alpha-{}.png'.format(method.lower(), num_components, alpha_value), dpi = 600, bbox_inches = 'tight' )

    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()

# C) PROGRAM:

# The Function to plot the Individual Histograms
def plot_individual_histograms(data_frame_transformed_extraction, method, num_components = 6, alpha_value = 0.8):

    # Initialise the Plot
    py_plot.figure( figsize = (15, 12), frameon = True )

    # Plot the Individual Histograms
    data_frame_transformed_extraction.hist( bins = 15, alpha = alpha_value, layout = (2, 3) )

    # Set the aspect of the Plot
    py_plot.gca().set_aspect( 'auto', adjustable = 'box' )

    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/data-analysis-visualization/{}-with-{}-components-data-individual-histograms-alpha-{}.png'.format(method.lower(), num_components, alpha_value), dpi = 600, bbox_inches = 'tight' )

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
    py_plot.savefig( 'imgs/plots/data-analysis-visualization/{}-with-{}-components-data-box.png'.format(method.lower(), num_components), dpi = 600, bbox_inches = 'tight' )

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
    py_plot.savefig('imgs/plots/data-analysis-visualization/{}-with-{}-components-data-scatter-matrix-{}-diagonal-alpha-{}.png'.format(method.lower(), num_components, diagonal_plot, alpha_value), dpi = 600, bbox_inches = 'tight' )

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
    py_plot.savefig( 'imgs/plots/data-analysis-visualization/{}-with-{}-components-data-parallel-coordinates.png'.format(method.lower(), num_components), dpi = 600, bbox_inches = 'tight' )

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
    pandas_plot.andrews_curves(data_frame_transformed_extraction, class_column = 'Celular Phase', color=("lightblue", "blue", "green", "brown"))

    # Set the Title of the Plot
    py_plot.title( 'Andrew\'s Curves, for {} with {} Components,\nrepresenting the different Classes in different Colours'.format(method, num_components) )

    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/data-analysis-visualization/{}-with-{}-components-data-andrew-curves.png'.format(method.lower(), num_components), dpi = 600, bbox_inches = 'tight' )

    # Adjust the Layout of the Plot to tight
    py_plot.tight_layout()

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()


def generate_analysis_plots(data_frame_transformed_extraction_pca, data_frame_columns_pca, data_frame_transformed_extraction_tsne, data_frame_columns_tsne, data_frame_transformed_extraction_isomap, data_frame_columns_isomap, num_components = 6):
    
    plot_stacked_histograms(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", num_components = num_components, alpha_value = 0.8)
    plot_stacked_histograms(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", num_components = num_components, alpha_value = 0.8)
    plot_stacked_histograms(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", num_components = num_components, alpha_value = 0.8)
    
    plot_individual_histograms(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", num_components = num_components, alpha_value = 0.8)
    plot_individual_histograms(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", num_components = num_components, alpha_value = 0.8)
    plot_individual_histograms(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", num_components = num_components, alpha_value = 0.8)
    
    plot_box(data_frame_transformed_extraction_pca[data_frame_columns_pca[0:-1]], "PCA", num_components = num_components)
    plot_box(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", num_components = num_components)
    plot_box(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", num_components = num_components)
    
    plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", "kde", num_components = num_components, alpha_value = 0.8)
    plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", "kde", num_components = num_components, alpha_value = 0.8)
    plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", "kde", num_components = num_components, alpha_value = 0.8)
    
    plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_pca[data_frame_columns_pca[0 : -1]], "PCA", "hist", num_components = num_components, alpha_value = 0.8)
    plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_tsne[data_frame_columns_tsne[0 : -1]], "TSNE", "hist", num_components = num_components, alpha_value = 0.8)
    plot_scatter_matrix_param_diagonal(data_frame_transformed_extraction_isomap[data_frame_columns_isomap[0 : -1]], "Isomap", "hist", num_components = num_components, alpha_value = 0.8)
    
    plot_parallel_coordinates(data_frame_transformed_extraction_pca, "PCA", num_components = num_components)
    plot_parallel_coordinates(data_frame_transformed_extraction_tsne, "TSNE", num_components = num_components)
    plot_parallel_coordinates(data_frame_transformed_extraction_isomap, "Isomap", num_components = num_components)
    
    plot_andrews_curves(data_frame_transformed_extraction_pca, "PCA", num_components = num_components)
    plot_andrews_curves(data_frame_transformed_extraction_tsne, "TSNE", num_components = num_components)
    plot_andrews_curves(data_frame_transformed_extraction_isomap, "Isomap", num_components = num_components)
    

def plot_elbow_method(errors_k_means_clustering, num_max_clusters = 10):
    
    # The xs data points ( Number of Clusters )
    xs_points = range( 1, (num_max_clusters + 1) )
    
    # The ys data points ( Errors (Sums of Squared Errors) )
    ys_points = errors_k_means_clustering
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Errors (Sums of Squared Errors) )
    py_plot.plot(xs_points, ys_points)
    
    # Set the Title of the Elbow Method Plot
    py_plot.title('Elbow Method for K-Means Clustering')
    
    # Set the label for the X axis of the Plot
    py_plot.xlabel('Number of Clusters')
    
    # Set the label for the Y axis of the Plot
    py_plot.ylabel('Errors (Sums of Squared Errors)')
        
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/elbow-method/elbow-method-k-means-clustering-for-max-of-{}-clusters.png'.format(num_max_clusters), dpi = 600, bbox_inches = 'tight' )

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()
    
    return xs_points, ys_points
    

def build_clusters_centroids_and_radii(xs_features_data, ys_labels_k_means_clusters, k_means_estimator_centroids):
    
    clusters_centroids = dict()
    clusters_radii = dict()
    
    for num_cluster in list(set(ys_labels_k_means_clusters)):
    
        clusters_centroids[num_cluster] = list(zip(k_means_estimator_centroids[:, 0], k_means_estimator_centroids[:,1]))[num_cluster]
        clusters_radii[num_cluster] = max([norm_number(subtract_numbers(data_point, clusters_centroids[num_cluster])) for data_point in zip(xs_features_data[ys_labels_k_means_clusters == num_cluster, 0], xs_features_data[ys_labels_k_means_clusters == num_cluster, 1])])

    return clusters_centroids, clusters_radii


def plot_clusters_centroids_and_radii(xs_features_data, ys_labels, k_means_estimator_centroids, num_clusters, final_clustering = False):
    
    figure, ax = py_plot.subplots(1, figsize = (12,8) )
    
    # Set the Title of the K-Means Clustering, for K Clusters
    py_plot.title( 'K-Means Clustering, for K = {} Cluster(s)'.format(num_clusters) )
    
    
    clusters_centroids, clusters_radii = build_clusters_centroids_and_radii(xs_features_data, ys_labels, k_means_estimator_centroids)
    
    
    for num_cluster in range(num_clusters):
    
        patch = matplotlib_patches.Circle(clusters_centroids[num_cluster], clusters_radii[num_cluster], edgecolor = 'black', facecolor = COLORS_MATPLOTLIB[num_cluster], fill = True, alpha = 0.125)
        ax.add_patch(patch)
        
        # Plot the Data (xs Points), as Scatter Points
        py_plot.scatter(xs_features_data[ys_labels == num_cluster, 0], xs_features_data[ys_labels == num_cluster, 1], color = COLORS_MATPLOTLIB[num_cluster], s = 100, label = "Cluster #{}".format(num_cluster))
        
        # Plot the Centroids of the Clusters, as Scatter Points
        py_plot.scatter(k_means_estimator_centroids[num_cluster][0], k_means_estimator_centroids[num_cluster, 1], marker = 'D', s = 200, color = 'black')
        
    
    
    # If it is the final K-Means Clustering,
    # with the best K found for the number of Clusters,
    # from the Elbow Method 
    if(final_clustering):
   
        # Save the Plot, as a figure/image
        py_plot.savefig( 'imgs/plots/final-k-means-clustering-centroids/k-means-clustering-for-{}-clusters-centroids.png'.format(num_clusters), dpi = 600, bbox_inches = 'tight' )

    # If it is varying the pre K-Means Clustering,
    # with the a certain K for the number of Clusters
    else:
        
        # Save the Plot, as a figure/image
        py_plot.savefig( 'imgs/plots/pre-k-means-clustering-centroids/k-means-clustering-for-{}-clusters-centroids.png'.format(num_clusters), dpi = 600, bbox_inches = 'tight' )
    

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()
    