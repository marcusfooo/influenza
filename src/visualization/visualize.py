import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def show_clusters(clusters, dims=2, method='TSNE'):
    for cluster in clusters:
        prot_vecs = cluster['prot_vecs']
        labels = cluster['labels']

        print(f'Shape: {np.array(prot_vecs).shape}')

        if(method == 'TSNE'):
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(prot_vecs)
            reduced_data = TSNE(random_state=8, n_components=dims).fit_transform(pca_result_50)
        if(method == 'PCA'):
            reduced_data = PCA(n_components=dims).fit_transform(prot_vecs)

        colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.']
        fig = plt.figure()
        for i in range(len(reduced_data)):
            if (dims == 2):
                plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)
        if (dims == 3):
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_data[:][0], reduced_data[:][1], reduced_data[:][2], colors[labels[i]], marker='o')

    plt.show()
