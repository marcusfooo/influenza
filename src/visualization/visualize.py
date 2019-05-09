import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import NearestNeighbors
import numpy as np

def show_clusters(clusters, dims=2, method='TSNE'):
    for cluster in clusters:
        prot_vecs = cluster['data']
        labels = cluster['labels']

        if(method == 'TSNE'):
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(prot_vecs)
            reduced_data = TSNE(random_state=8, n_components=dims).fit_transform(pca_result_50)
        if(method == 'PCA'):
            pca = PCA(n_components=dims)
            reduced_data = pca.fit_transform(prot_vecs)
            print(pca.explained_variance_ratio_)  
            reduced_centroids = NearestCentroid().fit(reduced_data, labels).centroids_

        fig = plt.figure()
        if (dims == 3): ax = fig.add_subplot(111, projection='3d')
        for i in range(len(reduced_data)):
            if (dims == 2):
                colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.']
                plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)
            if (dims == 3):
                colors = 10 * ['r', 'g', 'y', 'c', 'm', 'b', 'k']
                ax.scatter(reduced_data[i][0], reduced_data[i][1], reduced_data[i][2], c=colors[labels[i]], marker='.', zorder=1)
                centroid = reduced_centroids[labels[i]]
                ax.scatter(centroid[0], centroid[1], centroid[2], c='#0F0F0F', marker='x', zorder=100)

    plt.show()
