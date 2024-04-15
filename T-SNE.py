import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Carregar o dataset a partir do arquivo CSV
file_path = './datasets/Crop_recommendation.csv'
df = pd.read_csv(file_path)

# Selecionar as colunas relevantes para a análise
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]

# Aplicar o algoritmo k-means
kmeans = KMeans(n_clusters=8, random_state=42)  # Substitua 8 pelo número desejado de clusters
df['cluster'] = kmeans.fit_predict(features)

# Atualizar os centroides
centroids = kmeans.cluster_centers_

# Visualizar os resultados do K-means
plt.figure(figsize=(10, 8))
plt.scatter(df['temperature'], df['P'], c=df['cluster'], cmap='viridis', s=50, alpha=0.7)
plt.scatter(centroids[:, 3], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering of Environmental Data')
plt.xlabel('Temperature')
plt.ylabel('P')
plt.legend()
plt.show()

# Concatenar os rótulos de cluster como uma nova feature
features_with_cluster = pd.concat([features, pd.DataFrame({'cluster': df['cluster']})], axis=1)

# Aplicar o PCA aos dados
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

# Calcular os centroides após o PCA
centroids_pca = np.zeros((8, 2))
for cluster in range(8):
    centroids_pca[cluster] = np.mean(X_pca[df['cluster'] == cluster], axis=0)

# Salvar os resultados do PCA em um arquivo CSV
pca_df = pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2'])
pca_df.to_csv('PCA_results.csv', index=False)

# Visualizar os resultados do PCA
plt.figure(figsize=(10, 8))

# Definir uma lista de marcadores
markers = ['o', 's', '^', 'v', 'D', 'X', 'P', '*']

for cluster in range(8):  # Assumindo que existem 8 clusters, você pode ajustar conforme necessário
    cluster_points = X_pca[df['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f'Cluster {cluster}', s=50, alpha=0.7)
    # Selecionar apenas uma fração dos pontos para exibir os rótulos
    sample_indices = np.random.choice(len(cluster_points), min(10, len(cluster_points)), replace=False)
    for i, txt in enumerate(df[df['cluster'] == cluster].index[sample_indices]):
        plt.annotate(txt, (cluster_points[sample_indices[i], 0], cluster_points[sample_indices[i], 1]),
                     textcoords="offset points", xytext=(0, 5), ha='center')
    # Usar um marcador diferente para cada cluster
    plt.scatter(centroids_pca[cluster, 0], centroids_pca[cluster, 1], c='red', marker=markers[cluster], s=200, label=f'Centroid {cluster}')

plt.title('PCA Visualization with K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


# Aplicar o t-SNE aos dados, incluindo os rótulos de cluster como uma feature adicional
tsne = TSNE(n_components=2, learning_rate='auto', perplexity=5)
X_embedded = tsne.fit_transform(features_with_cluster)

# Calcular os centroides após o t-SNE
centroids_tsne = np.zeros((8, 2))
for cluster in range(8):
    centroids_tsne[cluster] = np.mean(X_embedded[df['cluster'] == cluster], axis=0)

# Salvar os resultados do t-SNE em um arquivo CSV
tsne_df = pd.DataFrame(X_embedded, columns=['t-SNE Dimension 1', 't-SNE Dimension 2'])
tsne_df.to_csv('tSNE_results.csv', index=False)

# Visualizar os resultados do t-SNE
plt.figure(figsize=(10, 8))

# Definir uma lista de marcadores
markers = ['o', 's', '^', 'v', 'D', 'X', 'P', '*']

for cluster in range(8):  # Assumindo que existem 8 clusters, você pode ajustar conforme necessário
    cluster_points = X_embedded[df['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f'Cluster {cluster}', s=50, alpha=0.7)
    # Selecionar apenas uma fração dos pontos para exibir os rótulos
    sample_indices = np.random.choice(len(cluster_points), min(10, len(cluster_points)), replace=False)
    for i, txt in enumerate(df[df['cluster'] == cluster].index[sample_indices]):
        plt.annotate(txt, (cluster_points[sample_indices[i], 0], cluster_points[sample_indices[i], 1]),
                     textcoords="offset points", xytext=(0, 5), ha='center')
    # Usar um marcador diferente para cada cluster
    plt.scatter(centroids_tsne[cluster, 0], centroids_tsne[cluster, 1], c='green', marker=markers[cluster], s=200, label=f'Centroid {cluster}')

plt.title('t-SNE Visualization with K-Means Clustering')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
