from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def cluster_years(trigram_vecs, method='DBSCAN'):
    clusters = []
    for year_trigram_vecs in trigram_vecs:
        prot_vecs = np.array(year_trigram_vecs).sum(axis=2)

        if(method == 'DBSCAN'):
            clf = DBSCAN(eps=0.01, metric='cosine').fit(prot_vecs)
            labels = clf.labels_

        if(method == 'KMeans'):
            clf = KMeans(n_clusters=3)
            clf.fit(prot_vecs)
            labels = clf.labels_

        clusters.append([prot_vecs, labels])

    return clusters

def evaluate_clusters(clusters):
    scores = []
    for cluster in clusters:
        prot_vecs = cluster[0]
        labels = cluster[1]

        score = silhouette_score(prot_vecs, labels)
        scores.append(score)

    average = sum(scores) / float(len(scores))
    print(f'Average score: {average}')