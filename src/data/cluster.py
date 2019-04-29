from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

import numpy as np
from math import floor

def cluster_years(prot_vecs, method='DBSCAN'):
    clusters = []
    for year_prot_vecs in prot_vecs:

        if(method == 'DBSCAN'):
            min_samples = floor(len(year_prot_vecs)*0.01)
            clf = DBSCAN(eps=5, min_samples=min_samples, metric='euclidean').fit(year_prot_vecs)
            labels = clf.labels_
            centroids = NearestCentroid().fit(year_prot_vecs, labels).centroids_

        if(method == 'MeanShift'):
            clf = MeanShift().fit(year_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        if(method == 'KMeans'):
            clf = KMeans(n_clusters=3)
            clf.fit(year_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        clusters.append({'prot_vecs':year_prot_vecs, 'labels':labels, 'centroids':centroids})

    return clusters

def squeeze_to_prot_vecs(trigram_vecs):
    prot_vecs = []
    for year_trigram_vecs in trigram_vecs:
        year_trigram_vecs = np.array(year_trigram_vecs).sum(axis=1)
        prot_vecs.append(year_trigram_vecs)
    return prot_vecs

def remove_outliers(clusters):
    for year_idx, cluster in enumerate(clusters):
        idxs_to_remove = []
        for i, label in enumerate(cluster['labels']):
            if(label == -1): idxs_to_remove.append(i)
        clusters[year_idx]['prot_vecs'] = [prot_vec for i, prot_vec in enumerate(cluster['prot_vecs']) if i not in idxs_to_remove]
        clusters[year_idx]['labels'] = [label for i, label in enumerate(cluster['labels']) if i not in idxs_to_remove]

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

def label_encode(strains_by_year):
    le = preprocessing.LabelEncoder()
    le.fit(list(strains_by_year[0][0]))

    encoded_strains = []
    for year_strains in strains_by_year:
        year_encoded_strains = []
        for strain in year_strains:
            chars = list(strain)
            year_encoded_strains.append(le.transform(chars))

        encoded_strains.append(year_encoded_strains)

    return encoded_strains

def cluster_raw(strains_by_year, processed_data, method='DBSCAN', metric='hamming'):
    clusters = []
    for i, year_strains in enumerate(strains_by_year):
        if(method == 'DBSCAN'):
            min_samples = floor(len(year_strains)*0.01)
            clf = DBSCAN(eps=0.03, min_samples=min_samples, metric=metric).fit(year_strains)
            labels = clf.labels_
            # centroids = NearestCentroid().fit(year_strains, labels).centroids_

        # clusters.append({'prot_vecs':processed_data[i], 'labels':labels, 'centroids':centroids})
        clusters.append({'prot_vecs':processed_data[i], 'labels':labels})


    return clusters