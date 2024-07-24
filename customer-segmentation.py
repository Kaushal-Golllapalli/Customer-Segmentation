import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')
data.head()

# Selecting features for segmentation
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method to determine the number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method graph
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans with the optimal number of clusters (let's assume it's 5)
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Adding cluster labels to the original dataset
data['Cluster'] = clusters

# Applying KMeans with the optimal number of clusters (let's assume it's 5)
kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Adding cluster labels to the original dataset
data['Cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(10,7))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Analyzing the segments
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features.columns)
cluster_centers_df['Cluster'] = range(0, len(cluster_centers_df))

print("Cluster Centers:")
print(cluster_centers_df)

# Descriptive statistics for each cluster
numeric_data = data.select_dtypes(include=[np.number])  # Selecting only numeric columns
numeric_data['Cluster'] = data['Cluster']  # Adding the cluster column back to numeric data
cluster_descriptions = numeric_data.groupby('Cluster').mean().reset_index()

print("Cluster Descriptions:")
print(cluster_descriptions)