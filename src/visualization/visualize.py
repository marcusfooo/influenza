import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def show_clusters(clusters, dims=2, method='TSNE'):
    for cluster in clusters:
        prot_vecs = cluster[0]
        labels = cluster[1]

        print(f'Shape: {np.array(prot_vecs).shape}')

        if(method == 'TSNE'):
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(prot_vecs)
            reduced_data = TSNE(random_state=8, n_components=dims).fit_transform(pca_result_50)
        if(method == 'PCA'):
            reduced_data = PCA(n_components=dims).fit_transform(prot_vecs)

        colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.']
        plt.figure()
        for i in range(len(reduced_data)):
            plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)

    plt.show()
