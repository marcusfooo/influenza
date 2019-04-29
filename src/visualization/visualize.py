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

        fig = plt.figure()
        if (dims == 3): ax = fig.add_subplot(111, projection='3d')
        for i in range(len(reduced_data)):
            if (dims == 2):
                colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.']
                plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)
            if (dims == 3):
                colors = 10 * ['r', 'g', 'y', 'c', 'm', 'b', 'k']
                ax.scatter(reduced_data[i][0], reduced_data[i][1], reduced_data[i][2], c=colors[labels[i]], marker='o')

    plt.show()
