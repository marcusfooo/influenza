from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def cluster_year(year_trigrams_vecs):
    numpi = np.array(year_trigrams_vecs)
    prot_vecs = numpi.sum(axis=2)
    print('Summed up to one vector per strain')
    print(f'Shape: {prot_vecs.shape}')

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(prot_vecs)

    print(pca_result[:5])
    print(f'Variance explain: {pca.explained_variance_ratio_}')

    clf = KMeans(n_clusters=3)
    clf.fit(pca_result)
    centroids = clf.cluster_centers_
    labels = clf.labels_

    print(centroids)
    print(labels)


    colors = ['g.', 'r.', 'c.', 'b.', 'k.']
    for i in range(len(pca_result)):
        plt.plot(pca_result[i][0], pca_result[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centroids[:,0], centroids[:,1],marker='X',s=150)

    plt.show()