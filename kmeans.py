import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar o dataset a partir do arquivo CSV
file_path = './Crop_recommendation.csv'
df = pd.read_csv(file_path)

# Selecionar as colunas relevantes para a análise
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]

# Aplicar o algoritmo k-means
kmeans = KMeans(n_clusters=8, random_state=42)  # Substitua 3 pelo número desejado de clusters
df['cluster'] = kmeans.fit_predict(features)

# Atualizar os centroides
centroids = kmeans.cluster_centers_

# Visualizar os resultados
plt.scatter(df['temperature'], df['P'], c=df['cluster'], cmap='viridis', s=50, alpha=0.7)
plt.scatter(centroids[:, 3], centroids[:, 1], c='red', marker='X', s=200, label='Updated Centroids')
plt.title('K-Means Clustering of Environmental Data with Updated Centroids')
plt.xlabel('Temperature')
plt.ylabel('P')
plt.legend()
plt.show()
