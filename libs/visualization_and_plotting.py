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

# -------------------------------------------------------------------#

# Import PyPlot Sub-Module,
# From Matplotlib Python's Library,
# as py_plot
from matplotlib import pyplot as py_plot

# Import Patches Sub-Module,
# From Matplotlib Python's Library,
# as matplotlib_patches
from matplotlib import patches as matplotlib_patches

# Import NanMax,
# From the NumPy's Python Library,
# as nan_max
from numpy import nanmax as nan_max

# Import Subtract Sub-Module,
# From NumPy Python's Library,
# as subtract_nums
from numpy import subtract as subtract_numbers

# Import arange,
# From the NumPy's Python Library,
# as a_range
from numpy import arange as a_range

# Import Array,
# From the NumPy's Python Library,
# as array_matrix
from numpy import array as array_matrix

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

# Import metrics.silhouette_samples Sub-Module,
# from SciKit-Learn Python's Library,
# as skl_silhouette_samples
from sklearn.metrics import silhouette_samples as skl_silhouette_samples



from libs.performance_scoring_metrics import compute_silhouette_score


COLORS_MATPLOTLIB = ['red', 'darkorange', 'goldenrod', 'yellow', 'lawngreen', 'forestgreen', 'turquoise', 'teal', 'deepskyblue', 'midnightblue', 'blue', 'darkviolet', 'magenta', 'pink', 'rosybrown', 'salmon', 'orangered', 'peru', 'burlywood', 'olive', 'springgreen', 'steelblue', 'slateblue', 'indigo', 'violet', 'deeppink']


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


def generate_data_analysis_plots(data_frame_transformed_extraction_pca, data_frame_columns_pca, data_frame_transformed_extraction_tsne, data_frame_columns_tsne, data_frame_transformed_extraction_isomap, data_frame_columns_isomap, num_components = 6):
    
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
    

def plot_elbow_method(clustering_algorithm, squared_errors_sums_intertias, best_num_clusters, num_max_clusters = 12):
    
    # The xs data points ( Number of Clusters )
    xs_points = array_matrix( range( 1, ( num_max_clusters + 1 ) ) )
    
    # The ys data points ( Errors (Sums of Squared Errors) )
    ys_points = squared_errors_sums_intertias
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Errors (Sums of Squared Errors) )
    py_plot.plot(xs_points, ys_points)
    
    py_plot.axvline(x = best_num_clusters, color = "black", linestyle = "--")
    
    # Set the Title of the Elbow Method Plot
    py_plot.title( 'Elbow Method for {} Clustering'.format(clustering_algorithm) )
    
    # Set the label for the X axis of the Plot
    py_plot.xlabel('Number of Clusters')
    
    # Set the label for the Y axis of the Plot
    py_plot.ylabel('Errors (Sums of Squared Errors)')
        
    # Show a Grid on the Plot
    py_plot.grid()
        
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/elbow-method/{}-clustering-elbow-method-for-max-of-{}-clusters.png'.format(clustering_algorithm.lower(), num_max_clusters), dpi = 600, bbox_inches = 'tight' )

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()

    
    return xs_points, ys_points, best_num_clusters



def plot_k_distance_method(clustering_algorithm, num_data_points_sorted_by_distance, k_neighbors_distances_epsilons, best_distance_epsilon):
    
    # The xs data points ( Number of Data Points )
    xs_points = num_data_points_sorted_by_distance
    
    # The ys data points ( K Neighbors' Distances / Epsilons )
    ys_points = k_neighbors_distances_epsilons
    
    # Plot the xs data points ( ε (Epsilons) ) and
    # their respective ys data points ( Distances )
    py_plot.plot(xs_points, ys_points)
    
    py_plot.axhline(y = best_distance_epsilon, color = "black", linestyle = "--")
        
    # Set the Title of the Elbow Method Plot
    py_plot.title( 'K-Distance Method for {} Clustering'.format(clustering_algorithm) )
    
    # Set the label for the X axis of the Plot
    py_plot.xlabel('Data Points Sorted by Distance')
    
    # Set the label for the Y axis of the Plot
    py_plot.ylabel('ε (Epsilon Value)')
        
    # Show a Grid on the Plot
    py_plot.grid()
    
    # Save the Plot, as a figure/image
    py_plot.savefig( 'imgs/plots/k-distance-method/{}-clustering-k-distance-method.png'.format(clustering_algorithm.lower()), dpi = 600, bbox_inches = 'tight' )

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()
    
    
    return xs_points, ys_points, best_distance_epsilon
    


def build_clusters_centroids_and_radii_k_means(xs_features_data, ys_labels_clusters, estimator_centroids):
    
    clusters_centroids = dict()
    clusters_radii = dict()
    
        
    for num_cluster in list(set(ys_labels_clusters)):
        
        if(num_cluster >= 0):
            
            clusters_centroids[num_cluster] = list(zip(estimator_centroids[:, 0], estimator_centroids[:, 1]))[int(num_cluster)]
    
            clusters_radii[num_cluster] = max([norm_number(subtract_numbers(data_point, clusters_centroids[num_cluster])) for data_point in zip(xs_features_data[ys_labels_clusters == num_cluster, 0], xs_features_data[ys_labels_clusters == num_cluster, 1])])


    return clusters_centroids, clusters_radii


def build_clusters_centroids_and_radii_dbscan_or_affinity_propagation(xs_features_data, ys_labels_predicted):
    
    num_clusters = ( nan_max(ys_labels_predicted) + 1 )
    current_cluster_i = 0
    
    clusters_centroids = None
    clusters_radii = None
    
    
    if(num_clusters > 0):
        
        clusters_centroids = matrix_array_zeros( (num_clusters, 2) )
        clusters_radii = matrix_array_zeros( (num_clusters) )
        
        for current_cluster_i in range(num_clusters):
            
            if(current_cluster_i >= 0):
                
                cluster_centroid_point_xs = xs_features_data[ys_labels_predicted == current_cluster_i, 0]
                cluster_centroid_point_ys = xs_features_data[ys_labels_predicted == current_cluster_i, 1]
                
                clusters_centroids[current_cluster_i, 0] = ( sum(cluster_centroid_point_xs) / len(cluster_centroid_point_xs) )
                clusters_centroids[current_cluster_i, 1] = ( sum(cluster_centroid_point_ys) / len(cluster_centroid_point_ys) )
                
                
                centroid_radii_max_value = 0
                
                
                for cluster_centroid_point in range( len(cluster_centroid_point_xs) ):
                    
                    distance_1 = pow(clusters_centroids[current_cluster_i, 0] - cluster_centroid_point_xs[cluster_centroid_point], 2)
                    distance_2 = pow(clusters_centroids[current_cluster_i, 1] - cluster_centroid_point_ys[cluster_centroid_point], 2)
                    
                    centroid_radii = pow( distance_1 + distance_2, ( 1/2 ) )
                    
                    
                    if(centroid_radii > centroid_radii_max_value):
                
                        centroid_radii_max_value = centroid_radii
                        clusters_radii[current_cluster_i] = centroid_radii_max_value
       
        
    return clusters_centroids, clusters_radii
    

def plot_clusters_centroids_and_radii(clustering_algorithm, xs_features_data, ys_labels_predicted, estimator_centroids, num_clusters, epsilon = None, damping = None, final_clustering = False):
     
    if( ( num_clusters > 0 ) and ( num_clusters <= 26 ) ):
    
        
        figure, ax = py_plot.subplots( 1, figsize = (12, 8) )
    
        py_plot.xlabel("Feature Space for the 1st Feature")
        py_plot.ylabel("Feature Space for the 2nd Feature")
         
        py_plot.xlim(-0.3, 1.5)
        py_plot.ylim(-0.5, 1.25)        
        
        
        if( ( clustering_algorithm == "DBSCAN" ) or ( clustering_algorithm == "Affinity-Propagation" ) ):
            
            clusters_centroids, clusters_radii = build_clusters_centroids_and_radii_dbscan_or_affinity_propagation(xs_features_data, ys_labels_predicted)
            
            
        else:
        
            clusters_centroids, clusters_radii = build_clusters_centroids_and_radii_k_means(xs_features_data, ys_labels_predicted, estimator_centroids)
            
    
        for current_cluster_i in list(set(ys_labels_predicted)):
        
            if(num_clusters <= 15):

                if(current_cluster_i != -1):
                    
                    patch_cluster_area = matplotlib_patches.Circle(clusters_centroids[int(current_cluster_i)], clusters_radii[int(current_cluster_i)], edgecolor = 'black', facecolor = COLORS_MATPLOTLIB[int(current_cluster_i)], fill = True, alpha = 0.125, label = "Cluster #{}".format(int(current_cluster_i)))
            
            else:
                
                if(current_cluster_i != -1):
                    
                    patch_cluster_area = matplotlib_patches.Circle(clusters_centroids[int(current_cluster_i)], clusters_radii[int(current_cluster_i)], edgecolor = 'black', facecolor = COLORS_MATPLOTLIB[int(current_cluster_i)], fill = True, alpha = 0.125)
            

            if(current_cluster_i != -1):
                    
                ax.add_patch(patch_cluster_area)
            
                # Plot the Data (xs Points), as Scatter Points
                py_plot.scatter(xs_features_data[ys_labels_predicted == int(current_cluster_i), 0], xs_features_data[ys_labels_predicted == int(current_cluster_i), 1], color = COLORS_MATPLOTLIB[int(current_cluster_i)], s = 20, label = "Points in Cluster #{}".format(int(current_cluster_i)))
            
            
            if( ( clustering_algorithm == "DBSCAN" ) or ( clustering_algorithm == "Affinity-Propagation" ) ):

                # Plot the Centroids of the Clusters, as Scatter Points
                py_plot.scatter(clusters_centroids[int(current_cluster_i)][0], clusters_centroids[int(current_cluster_i), 1], marker = 'D', s = 100, color = 'dimgray')       
                
                
            else:

                # Plot the Centroids of the Clusters, as Scatter Points
                py_plot.scatter(estimator_centroids[int(current_cluster_i)][0], estimator_centroids[int(current_cluster_i), 1], marker = 'D', s = 100, color = 'dimgray')
            
            
        if(clustering_algorithm == "DBSCAN"):
    
            # Plot the Data (xs Points), related to noise (outliers), as Scatter Points
            py_plot.scatter(xs_features_data[ys_labels_predicted == -1, 0], xs_features_data[ys_labels_predicted == -1, 1], color = 'black', s = 20, label = "Outliers (Noise Points)")
            
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc = "upper right", frameon = True)
        
        
        # Show a Grid on the Plot
        ax.grid()
        
        
        # If it is the final K-Means Clustering,
        # with the best K found for the number of Clusters,
        # from the Elbow Method 
        if(final_clustering):
    
            if( ( clustering_algorithm == "DBSCAN" ) and ( epsilon != None ) ):
       
                # Set the Title of the DBSCAN Clustering, for an ε (Epsilon Value)
                py_plot.title( 'Final/Best {} Clustering, with K = {} Cluster(s) and ε (Epsilon Value) = {}'.format(clustering_algorithm, num_clusters, epsilon) )           
            
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/final-{}-clustering-centroids/final-{}-clustering-for-{}-clusters-centroids-and-epsilon-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, epsilon), dpi = 600, bbox_inches = 'tight' )
    
            
            elif( ( clustering_algorithm == "Affinity-Propagation" ) and ( damping != None ) ):
       
                # Set the Title of the Affinity Propagation Clustering, for a γ (Damping Value)
                py_plot.title( 'Final/Best {} Clustering, with K = {} Cluster(s) and γ (Damping Value) = {}'.format(clustering_algorithm, num_clusters, damping) )           
            
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/final-{}-clustering-centroids/final-{}-clustering-for-{}-clusters-centroids-and-damping-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, damping), dpi = 600, bbox_inches = 'tight' )
                
                
            elif( clustering_algorithm == "Bisecting-K-Means" ):
   
                # Set the Title of the Bisecting K-Means Clustering, for K Clusters
                py_plot.title( 'Final/Best {} Clustering, with K = {} Cluster(s)'.format(clustering_algorithm, num_clusters) )           
            
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/final-{}-clustering-centroids/final-{}-clustering-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )
            
            
            else:
                    
                # Set the Title of the K-Means Clustering, for K Clusters
                py_plot.title( 'Final/Best {} Clustering, with K = {} Cluster(s)'.format(clustering_algorithm, num_clusters) )           
            
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/final-{}-clustering-centroids/final-{}-clustering-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )
    
        
        # If it is varying the pre K-Means Clustering,
        # with the a certain K for the number of Clusters
        else:
       
            if( ( clustering_algorithm == "DBSCAN" ) and ( epsilon != None ) ):
       
                # Set the Title of the DBSCAN Clustering, for an ε (Epsilon Value)
                py_plot.title( '{} Clustering, with K = {} Cluster(s) and ε (Epsilon Value) = {}'.format(clustering_algorithm, num_clusters, epsilon) )     
                
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/pre-{}-clustering-centroids/pre-{}-clustering-for-{}-clusters-centroids-and-epsilon-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, epsilon), dpi = 600, bbox_inches = 'tight' )
              
                
            elif( ( clustering_algorithm == "Affinity-Propagation" ) and ( damping != None ) ):
       
                # Set the Title of the Affinity-Propagation Clustering, for a γ (Damping Value)
                py_plot.title( '{} Clustering, with K = {} Cluster(s) and γ (Damping Value) = {}'.format(clustering_algorithm, num_clusters, damping) )     
                
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/pre-{}-clustering-centroids/pre-{}-clustering-for-{}-clusters-centroids-and-damping-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, damping), dpi = 600, bbox_inches = 'tight' )
            
                
            elif( clustering_algorithm == "Bisecting-K-Means" ):
   
                # Set the Title of the Bisecting K-Means Clustering, for K Clusters
                py_plot.title( '{} Clustering, with K = {} Cluster(s)'.format(clustering_algorithm, num_clusters) )           
            
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/pre-{}-clustering-centroids/pre-{}-clustering-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )
        
       
            else:
        
                # Set the Title of the K-Means Clustering, for K Clusters
                py_plot.title( '{} Clustering, with K = {} Cluster(s)'.format(clustering_algorithm, num_clusters) )     
                
                # Save the Plot, as a figure/image
                py_plot.savefig( 'imgs/plots/pre-{}-clustering-centroids/pre-{}-clustering-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )
        
    
        # Show the Plot
        py_plot.show()
    
        # Close the Plot
        py_plot.close()


def plot_silhouette_analysis(clustering_algorithm, xs_features_data, ys_labels_predicted, estimator_centroids, num_clusters, epsilon = None, damping = None, final_clustering = False):
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = py_plot.subplots(1, 2)

    # Set the Size's Inches for the Figure
    fig.set_size_inches(24, 8)        
    
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(xs_features_data) + (num_clusters + 1) * 10])
    
    ax2.set_xlim(-0.3, 1.5)
    ax2.set_ylim(-0.5, 1.25)  

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_score_average = compute_silhouette_score(xs_features_data, ys_labels_predicted)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = skl_silhouette_samples(xs_features_data, ys_labels_predicted)
    
    
    if( ( clustering_algorithm == "DBSCAN" ) or ( clustering_algorithm == "Affinity-Propagation" ) ):
        
        clusters_centroids, clusters_radii = build_clusters_centroids_and_radii_dbscan_or_affinity_propagation(xs_features_data, ys_labels_predicted)
        
        
    else:
        
        clusters_centroids, clusters_radii = build_clusters_centroids_and_radii_k_means(xs_features_data, ys_labels_predicted, estimator_centroids)
        

    
    y_lower = 10
    
    
    for current_cluster_i in list(set(ys_labels_predicted)):
            
        if(num_clusters <= 15):
        
            if(current_cluster_i != -1):
                    
                patch_cluster_area = matplotlib_patches.Circle(clusters_centroids[int(current_cluster_i)], clusters_radii[int(current_cluster_i)], edgecolor = 'black', facecolor = COLORS_MATPLOTLIB[int(current_cluster_i)], fill = True, alpha = 0.125, label = "Cluster #{}".format(int(current_cluster_i)))    
        
        else:
        
            if(current_cluster_i != -1):
                    
                patch_cluster_area = matplotlib_patches.Circle(clusters_centroids[int(current_cluster_i)], clusters_radii[int(current_cluster_i)], edgecolor = 'black', facecolor = COLORS_MATPLOTLIB[int(current_cluster_i)], fill = True, alpha = 0.125)    
        
        
        if(current_cluster_i != -1):
                    
            ax2.add_patch(patch_cluster_area)
        
            ax2.scatter(xs_features_data[ys_labels_predicted == int(current_cluster_i), 0], xs_features_data[ys_labels_predicted == int(current_cluster_i), 1], color = COLORS_MATPLOTLIB[int(current_cluster_i)], s = 30, label = "Points in Cluster #{}".format(int(current_cluster_i)))


        if(clustering_algorithm == "DBSCAN"):
            
            if(int(current_cluster_i) == ( num_clusters - 1 ) ):
                    
                ax2.scatter(xs_features_data[ys_labels_predicted == -1, 0], xs_features_data[ys_labels_predicted == -1, 1], color = "black", s = 30, label = "Outliers (Noise Points)")
            
            else:
                
                ax2.scatter(xs_features_data[ys_labels_predicted == -1, 0], xs_features_data[ys_labels_predicted == -1, 1], color = "black", s = 30)    
            
    
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[ys_labels_predicted == int(current_cluster_i)]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
    
        y_upper = y_lower + size_cluster_i

        if(int(current_cluster_i) != -1): 
        
            color = COLORS_MATPLOTLIB[int(current_cluster_i)]
                   
            ax1.fill_betweenx(a_range(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor = color, edgecolor = color, alpha = 0.75)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text( -0.05, ( y_lower + ( 0.5 * size_cluster_i ) ), str(int(current_cluster_i)) )

        # Compute the new y_lower for next plot
        y_lower = ( y_upper + 10 ) # 10 for the 0 samples

    ax1.set_title("The Silhouette Plot for the several Clusters")
    ax1.set_xlabel("The Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster Number (Label)")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x = silhouette_score_average, color = "black", linestyle = "--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])    
    
    
    if( ( clustering_algorithm == "DBSCAN" ) or ( clustering_algorithm == "Affinity-Propagation" ) ):
        
        for current_cluster_i in range(num_clusters):
            
            # Draw white circles at cluster centers
            ax2.scatter( clusters_centroids[current_cluster_i][0], clusters_centroids[current_cluster_i, 1], marker = 'o', c = "white", alpha = 1, s = 200, edgecolor = 'black' )
        
            ax2.scatter( clusters_centroids[current_cluster_i][0], clusters_centroids[current_cluster_i, 1], marker = ( '$%d$' % current_cluster_i ), alpha = 1, s = 50, edgecolor = 'black' )
        
                    
    else:
        
        if( clustering_algorithm == "Bisecting-K-Means" ):

            cluster_labels_set_list = list(set(ys_labels_predicted))      

            for current_cluster_i in cluster_labels_set_list:
                
                # Draw white circles at cluster centers
                ax2.scatter( estimator_centroids[int(current_cluster_i), 0], estimator_centroids[int(current_cluster_i), 1], marker = 'o', c = "white", alpha = 1, s = 200, edgecolor = 'black' )
            
            for num_cluster_centroid, cluster_centroid in enumerate(estimator_centroids):
                
                if(num_cluster_centroid in cluster_labels_set_list):
                
                    ax2.scatter( cluster_centroid[0], cluster_centroid[1], marker = ( '$%d$' % num_cluster_centroid ), alpha = 1, s = 50, edgecolor = 'black' )

            
        else:
            
            # Draw white circles at cluster centers
            ax2.scatter( estimator_centroids[:, 0], estimator_centroids[:, 1], marker = 'o', c = "white", alpha = 1, s = 200, edgecolor = 'black' )
        
            for num_cluster_centroid, cluster_centroid in enumerate(estimator_centroids):
                
                ax2.scatter( cluster_centroid[0], cluster_centroid[1], marker = ( '$%d$' % num_cluster_centroid ), alpha = 1, s = 50, edgecolor = 'black' )

    
    ax2.set_title("The Visualization of the Clustered Data")
    ax2.set_xlabel("Feature Space for the 1st Feature")
    ax2.set_ylabel("Feature Space for the 2nd Feature")
    
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc = "upper right", frameon = True)
    
    # Show Grids on the Plots
    ax1.grid()
    ax2.grid()


    if(final_clustering):

        if( ( clustering_algorithm == "DBSCAN" ) and ( epsilon != None ) ):
            
            py_plot.suptitle( "Final/Best Silhouette Analysis for {} Clustering, on Sample Data, with K = {} Cluster(s) and ε (Epsilon Value) = {}".format(clustering_algorithm, num_clusters, epsilon), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/final-{}-clustering-silhouette-analysis/final-{}-clustering-silhouette-analysis-for-{}-clusters-centroid-and-epsilon-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, epsilon), dpi = 600, bbox_inches = 'tight' )
                        
            
        elif( ( clustering_algorithm == "Affinity-Propagation" ) and ( damping != None ) ):
            
            py_plot.suptitle( "Final/Best Silhouette Analysis for {} Clustering, on Sample Data, with K = {} Cluster(s) and γ (Damping Value) = {}".format(clustering_algorithm, num_clusters, damping), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/final-{}-clustering-silhouette-analysis/final-{}-clustering-silhouette-analysis-for-{}-clusters-centroid-and-damping-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, damping), dpi = 600, bbox_inches = 'tight' )
                        
            
        else:

            py_plot.suptitle( "Final/Best Silhouette Analysis for {} Clustering, on Sample Data, with K = {} Cluster(s)".format(clustering_algorithm, num_clusters), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/final-{}-clustering-silhouette-analysis/final-{}-clustering-silhouette-analysis-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )

        
    else:
        
        if( ( clustering_algorithm == "DBSCAN" ) and ( epsilon != None ) ):

            py_plot.suptitle( "Silhouette Analysis for {} Clustering, on Sample Data, with K = {} Cluster(s) and ε (Epsilon Value) = {}".format(clustering_algorithm, num_clusters, epsilon), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/pre-{}-clustering-silhouette-analysis/pre-{}-clustering-silhouette-analysis-for-{}-clusters-centroids-and-epsilon-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, epsilon), dpi = 600, bbox_inches = 'tight' )
            
            
        elif( ( clustering_algorithm == "Affinity-Propagation" ) and ( damping != None ) ):

            py_plot.suptitle( "Silhouette Analysis for {} Clustering, on Sample Data, with K = {} Cluster(s) and γ (Damping Value) = {}".format(clustering_algorithm, num_clusters, damping), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/pre-{}-clustering-silhouette-analysis/pre-{}-clustering-silhouette-analysis-for-{}-clusters-centroids-and-damping-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, damping), dpi = 600, bbox_inches = 'tight' )
            
            
        else:
                
            py_plot.suptitle( "Silhouette Analysis for {} Clustering, on Sample Data, with K = {} Cluster(s)".format(clustering_algorithm, num_clusters), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/pre-{}-clustering-silhouette-analysis/pre-{}-clustering-silhouette-analysis-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )
    

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()
    
    
    
def plot_confusion_matrix_rand_index_clustering_heatmap(clustering_algorithm, confusion_matrix_rand_index_clustering, num_clusters, epsilon = None, damping = None, final_clustering = False):
    
    groups_labels = ["Same Group", "Different Group"]
    clusters_labels = ["Same Cluster", "Different Cluster"]


    confusion_matrix_rand_index_clustering = array_matrix([[confusion_matrix_rand_index_clustering[0][0], confusion_matrix_rand_index_clustering[0][1]],
                                                           [confusion_matrix_rand_index_clustering[1][0], confusion_matrix_rand_index_clustering[1][1]]])
    
    fig, ax = py_plot.subplots()
    ax.imshow(confusion_matrix_rand_index_clustering, cmap = "PuOr")
    
    # We want to show all ticks
    ax.set_xticks(a_range(len(groups_labels)))
    ax.set_yticks(a_range(len(clusters_labels)))
    
    # And label them with the respective list entries
    ax.set_xticklabels(groups_labels)
    ax.set_yticklabels(clusters_labels)
    
    # Rotate the tick labels and set their alignment
    py_plot.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    
    # Loop over data dimensions and create text annotations
    for group_label in range(len(groups_labels)):
        
        for cluster_label in range(len(clusters_labels)):
            
            ax.text(cluster_label, group_label, confusion_matrix_rand_index_clustering[cluster_label, group_label], ha = "center", va = "center", color = "w")
    
    
    # Set a Tight Layout for the Plot
    fig.tight_layout()
    
    
    if(final_clustering):

        if( ( clustering_algorithm == "DBSCAN" ) and ( epsilon != None ) ):
            
            ax.set_title( "Final/Best Heatmap for {} Clustering, on Sample Data, with K = {} Cluster(s) and ε (Epsilon Value) = {}".format(clustering_algorithm, num_clusters, epsilon), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/final-{}-clustering-heatmaps/final-{}-clustering-heatmap-for-{}-clusters-centroids-and-epsilon-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, epsilon), dpi = 600, bbox_inches = 'tight' )
            
        
        elif( ( clustering_algorithm == "Affinity-Propagation" ) and ( damping != None ) ):
            
            ax.set_title( "Final/Best Heatmap for {} Clustering, on Sample Data, with K = {} Cluster(s) and γ (Damping Value) = {}".format(clustering_algorithm, num_clusters, damping), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/final-{}-clustering-heatmaps/final-{}-clustering-heatmap-for-{}-clusters-centroids-and-damping-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, damping), dpi = 600, bbox_inches = 'tight' )


        else:
                
            ax.set_title( "Final/Best Heatmap for {} Clustering, on Sample Data, with K = {} Cluster(s)".format(clustering_algorithm, num_clusters), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/final-{}-clustering-heatmaps/final-{}-clustering-heatmap-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )

        
    else:

        if( ( clustering_algorithm == "DBSCAN" ) and ( epsilon != None ) ):
            
            ax.set_title( "Heatmap for {} Clustering, on Sample Data, with K = {} Cluster(s) and ε (Epsilon Value) = {}".format(clustering_algorithm, num_clusters, epsilon), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/pre-{}-clustering-heatmaps/pre-{}-clustering-heatmap-for-{}-clusters-centroids-and-epsilon-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, epsilon), dpi = 600, bbox_inches = 'tight' )
        
        
        if( ( clustering_algorithm == "Affinity-Propagation" ) and ( damping != None ) ):
            
            ax.set_title( "Heatmap for {} Clustering, on Sample Data, with K = {} Cluster(s) and γ (Damping Value) = {}".format(clustering_algorithm, num_clusters, damping), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/pre-{}-clustering-heatmaps/pre-{}-clustering-heatmap-for-{}-clusters-centroids-and-damping-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters, damping), dpi = 600, bbox_inches = 'tight' )
        
    
        else:
            
            ax.set_title( "Heatmap for {} Clustering, on Sample Data, with K = {} Cluster(s)".format(clustering_algorithm, num_clusters), fontsize = 14, fontweight = 'bold' )
    
            # Save the Plot, as a figure/image
            py_plot.savefig( 'imgs/plots/pre-{}-clustering-heatmaps/pre-{}-clustering-heatmap-for-{}-clusters-centroids.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_clusters), dpi = 600, bbox_inches = 'tight' )
    

    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()
    
    
    
def plot_clustering_scores(clustering_algorithm, num_total_clusters, start_epsilon, end_epsilon, step_epsilon, epsilon_values, start_damping, end_damping, step_damping, damping_values, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores):
    
    if(clustering_algorithm == "DBSCAN"):
        
        # The xs data points ( Number of Clusters )
        xs_points = a_range(start_epsilon, end_epsilon, step_epsilon)
    
    elif(clustering_algorithm == "Affinity-Propagation"):
        
        # The xs data points ( Number of Clusters )
        xs_points = a_range(start_damping, end_damping, step_damping)
        
    else:
        
        # The xs data points ( Number of Clusters )
        xs_points = range( 2, ( num_total_clusters + 1 ) )
    
    
    # The ys data points ( Silhouette Scores )
    ys_points_silhouette_scores = clusters_silhouette_scores
    
    # The ys data points ( Precision Scores )
    ys_points_precision_scores = clusters_precision_scores
    
    # The ys data points ( Recall Scores )
    ys_points_recall_scores = clusters_recall_scores
    
    # The ys data points ( Rand Index Scores )
    ys_points_rand_index_scores = clusters_rand_index_scores
    
    # The ys data points ( F1 Scores )
    ys_points_f1_scores = clusters_f1_scores
    
    # The ys data points ( Adjusted Rand Scores )
    ys_points_adjusted_rand_scores = clusters_adjusted_rand_scores
    
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Silhouette Scores )
    py_plot.plot(xs_points, ys_points_silhouette_scores, color = "red", label = "Silhouette Scores")
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Precision Scores )
    py_plot.plot(xs_points, ys_points_precision_scores, color = "green", label = "Precison Scores")
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Recall Scores )
    py_plot.plot(xs_points, ys_points_recall_scores, color = "blue", label = "Recall Scores")
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Rand Index Scores )
    py_plot.plot(xs_points, ys_points_rand_index_scores, color = "orange", label = "Rand Index Scores")
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( F1 Scores )
    py_plot.plot(xs_points, ys_points_f1_scores, color = "purple", label = "F1 Scores")
    
    # Plot the xs data points ( Number of Clusters ) and
    # their respective ys data points ( Adjusted Rand Scores )
    py_plot.plot(xs_points, ys_points_adjusted_rand_scores, color = "dimgray", label = "Adjusted Rand Scores")
    
    
    # Place the Labels (Legend) on the Upper Right corner
    py_plot.legend(loc = "upper right", frameon = True)
    
    
    # Show a Grid on the Plot
    py_plot.grid()


    # Set the Title of the Performance Metrics/Scores Plot
    py_plot.title( 'Performance Metrics/Scores for {} Clustering'.format(clustering_algorithm) )
    
    
    if(clustering_algorithm == "DBSCAN"):
        
        # Set the label for the X axis of the Plot
        py_plot.xlabel('ε (Epsilon Value)')
        
    elif(clustering_algorithm == "Affinity-Propagation"):
        
        # Set the label for the X axis of the Plot
        py_plot.xlabel('γ (Damping Value)')
        
    else:
        
        # Set the label for the X axis of the Plot
        py_plot.xlabel('Number of Clusters')
        
    
    # Set the label for the Y axis of the Plot
    py_plot.ylabel('Performance Metrics/Scores')

        
    if(clustering_algorithm == "DBSCAN"):
        
        # Save the Plot, as a figure/image
        py_plot.savefig( 'imgs/plots/{}-clustering-performance-metrics-scores/{}-clustering-performance-metrics-scores-for-max-of-epsilon-value-of-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), end_epsilon), dpi = 600, bbox_inches = 'tight' )

    elif(clustering_algorithm == "Affinity-Propagation"):
        
        # Save the Plot, as a figure/image
        py_plot.savefig( 'imgs/plots/{}-clustering-performance-metrics-scores/{}-clustering-performance-metrics-scores-for-max-of-damping-value-of-{}.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), end_damping), dpi = 600, bbox_inches = 'tight' )
                
    else:
        
        # Save the Plot, as a figure/image
        py_plot.savefig( 'imgs/plots/{}-clustering-performance-metrics-scores/{}-clustering-performance-metrics-scores-for-max-of-{}-clusters.png'.format(clustering_algorithm.lower(), clustering_algorithm.lower(), num_total_clusters), dpi = 600, bbox_inches = 'tight' )


    # Show the Plot
    py_plot.show()

    # Close the Plot
    py_plot.close()

    
    return xs_points, ys_points_silhouette_scores, ys_points_precision_scores, ys_points_recall_scores, ys_points_rand_index_scores, ys_points_f1_scores, ys_points_adjusted_rand_scores