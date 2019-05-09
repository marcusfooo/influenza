from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import math
import random

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

        clusters.append({'data':year_prot_vecs, 'labels':labels, 'centroids':centroids})

    return clusters

def squeeze_to_prot_vecs(trigram_vecs):
    prot_vecs = []
    for year_trigram_vecs in trigram_vecs:
        year_trigram_vecs = np.array(year_trigram_vecs).sum(axis=1)
        prot_vecs.append(year_trigram_vecs)
    return prot_vecs

def remove_outliers(data, clusters):
    for year_idx, cluster in enumerate(clusters):
        idxs_to_remove = []
        for i, label in enumerate(cluster['labels']):
            if(label == -1): idxs_to_remove.append(i) # -1 means outlier
        data[year_idx] = [prot_vec for i, prot_vec in enumerate(data[year_idx]) if i not in idxs_to_remove]
        clusters[year_idx]['labels'] = [label for i, label in enumerate(cluster['labels']) if i not in idxs_to_remove]
        clusters[year_idx]['data'] = [strain for i, strain in enumerate(cluster['data']) if i not in idxs_to_remove]
        if -1 in clusters[year_idx]['population']: del clusters[year_idx]['population'][-1]

    return data, clusters
        
def evaluate_clusters(clusters):
    scores = []
    for cluster in clusters:
        score = silhouette_score(cluster['data'], cluster['labels'])
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
            if (idxs_by_centroid[label][0] == -1): del idxs_by_centroid[label][0]
            links.append(idxs_by_centroid[label]) # centroid idx corresponds to label

        clusters[year_idx]['links'] = links

    return clusters

def label_encode(strains_by_year):
    amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L']
    le = preprocessing.LabelEncoder()
    le.fit(amino_acids)

    encoded_strains = []
    for year_strains in strains_by_year:
        year_encoded_strains = []
        for strain in year_strains:
            chars = list(strain)
            year_encoded_strains.append(le.transform(chars))

        encoded_strains.append(year_encoded_strains)

    return encoded_strains

def cluster_raw(strains_by_year, method='DBSCAN'):
    clusters = []
    for year_strains in strains_by_year:
        if(method == 'DBSCAN'):
            min_samples = floor(len(year_strains)*0.02)
            clf = DBSCAN(eps=0.01, min_samples=min_samples, metric='hamming').fit(year_strains)
            # clf = DBSCAN(eps=10, min_samples=5, metric=metric).fit(year_strains)
            labels = clf.labels_

        unique, count = np.unique(labels, return_counts=True)

        cluster = {
            'data': year_strains,
            'labels':labels, 
            'population': dict(zip(unique, count))} 

        clusters.append(cluster)
    return clusters

def sample_from_clusters(strains_by_year, clusters_by_years, sample_size):
    sampled_strains = [[]] * len(strains_by_year)

    # start sample from first cluster
    first_year_labels = clusters_by_years[0]['labels']
    first_year_population = clusters_by_years[0]['population']
    first_year_total = len(strains_by_year[0])

    for label_idx in range(len(first_year_population)):
        cluster_proportion = first_year_population[label_idx]/first_year_total
        cluster_sample_size = math.floor(sample_size*cluster_proportion)
        cluster_strains = [strains_by_year[0][i] for i, label in enumerate(first_year_labels) if label == label_idx]
        sampled_strains[0] = sampled_strains[0] + random.choices(cluster_strains, k=cluster_sample_size)

    # sample forward
    current_cluster = label_encode([sampled_strains[0]])[0]
    for year_idx, year_clusters in enumerate(clusters_by_years[1:]):
        neigh = NearestNeighbors(n_neighbors=1, metric='hamming')
        neigh.fit(year_clusters['data'])
        neighbour_strain_idx = neigh.kneighbors(current_cluster, return_distance=False)

        links = [year_clusters['labels'][idx[0]] for idx in neighbour_strain_idx]

        clustered_strains = []
        for label_idx in range(len(year_clusters['population'])):
           clustered_strains.append([strains_by_year[year_idx+1][i] for i, label in enumerate(year_clusters['labels']) if label == label_idx])

        for link in links:
           sampled_strains[year_idx+1].append(random.choice(clustered_strains[link]))

    return sampled_strains
            

          
