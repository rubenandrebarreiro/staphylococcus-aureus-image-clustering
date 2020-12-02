# -*- coding: utf-8 -*-
"""

@author: Martim Figueiredo - 42648
@author: Rúben André Barreiro - 42648

"""

# Libraries:

# - 1) General Libraries:

# Import loadtxt,
# From the NumPy's Python Library,
# as load_txt
from numpy import loadtxt as load_txt

from numpy import savetxt as save_txt



# - 2) Libraries for Features Extraction:

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


# - 3) Customised Libraries:

# Import images_as_matrix,
# From the TP2_Aux Custom Python Library,
# as images_as_matrix
from tp2_aux import images_as_matrix as images_as_numpy_matrix


# The 2D NumPy Matrix, representing all the samples images,
# with an image per row (563 lines),
# and one indiviual pixel by column
# (50 pixels x 50 pixels = 2500 columns)
xs_images_matrix = images_as_numpy_matrix(N = 563)

# Just for Debug
# save_txt(fname="images_matrix.txt", X=xs_images_as_numpy_matrix, delimiter=", ", newline="\n")

ids_and_labels = load_txt("labels.txt", delimiter=",")


# Just for Debug
# print(ids_and_labels)

# The Labels for Celular Phases of the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
ys_labels = ids_and_labels[:,1]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
xs_ids = ids_and_labels[:,0]


# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies,
# which are not labelled
ids_not_labelled_data = ids_and_labels[ys_labels == 0]

# The Identifiers for the samples of Staphycoccus Aureus,
# provided by Super-Resolution Fluorescence Microscopy Photographies
# which are labelled
ids_labelled_data = ids_and_labels[ys_labels != 0]



# For each method of Features Extraction
# (3 methods of Extraction of Features),
# extract 6 Features (a total of 3 x 6 = 18 features):
# - 1) PCA (Principal Component Analysis) Decomposition;
# - 2) TSNE (T-Distributed Stochastic Neighbor Embedding);
# - 3) Isomap (Isometric Mapping);
pca = pca_decomposition(n_components = 6)


# Fit the PCA (Principal Component Analysis) Decomposition,
# with the 2D NumPy Matrix, representing all the images
pca.fit(xs_images_matrix)


transformed_xs_images_matrix = pca.transform(xs_images_matrix)


