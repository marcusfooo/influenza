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
    no_years = len(clusters)
    neigh = NearestNeighbors(n_neighbors=2)

    for year_idx in range(no_years): 
        if(year_idx == no_years-1): # last year doesn't link
            clusters[year_idx]['links'] = [] 
            break 

        links = []
        current_centroids = clusters[year_idx]['centroids']
        next_year_centroids = clusters[year_idx+1]['centroids']
        neigh.fit(next_year_centroids)

        idxs_by_centroid = neigh.kneighbors(current_centroids, return_distance=False)

        for label in clusters[year_idx]['labels']:
            links.append(idxs_by_centroid[label]) # centroid idx corresponds to label

        clusters[year_idx]['links'] = links

    return clusters


