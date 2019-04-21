from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import NearestNeighbors

import numpy as np
from math import floor

def cluster_years(trigram_vecs, method='DBSCAN', remove_outliers=False):
    clusters = []
    for year_trigram_vecs in trigram_vecs:
        prot_vecs = np.array(year_trigram_vecs).sum(axis=1)

        if(method == 'DBSCAN'):
            min_samples = floor(len(year_trigram_vecs)*0.01)
            clf = DBSCAN(eps=5, min_samples=min_samples, metric='euclidean').fit(prot_vecs)
            labels = clf.labels_
            centroids = NearestCentroid().fit(prot_vecs, labels).centroids_

        if(method == 'MeanShift'):
            clf = MeanShift().fit(prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        if(method == 'KMeans'):
            clf = KMeans(n_clusters=3)
            clf.fit(prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        if(remove_outliers):
            idxs_to_remove = []
            for i, label in enumerate(labels):
                if(label == -1): idxs_to_remove.append(i)
            prot_vecs = [prot_vec for i, prot_vec in enumerate(prot_vecs) if i not in idxs_to_remove]
            labels = [label for i, label in enumerate(labels) if i not in idxs_to_remove]

        clusters.append({'prot_vecs':prot_vecs, 'labels':labels, 'centroids':centroids})

    return clusters

def evaluate_clusters(clusters):
    scores = []
    for cluster in clusters:
        score = silhouette_score(cluster['prot_vecs'], cluster['labels'])
        scores.append(score)

    average = sum(scores) / float(len(scores))
    return average

def link_clusters(clusters):
    linked = []
    no_years = len(clusters)

    first_cluster = clusters[0]
    centroids = first_cluster['centroids']

    neigh = NearestNeighbors(n_neighbors=1)

    for i, centroid in enumerate(centroids):
        links = []
        current_centroid = centroid

        for year_idx in range(no_years-1):
            year_centroids = clusters[year_idx+1]['centroids']
            neigh.fit(year_centroids)

            ind = neigh.kneighbors([current_centroid], return_distance=False)
            current_centroid = year_centroids[ind[0][0]]
            links.append(ind)
        
        linked.append(links)

    return linked