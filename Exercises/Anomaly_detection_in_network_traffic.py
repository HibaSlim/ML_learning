import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('KDDCup99.csv')
data = df[['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate']]
data = data.drop_duplicates()

#Normalize the data to respect the normal distribution
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#K-means implementation
sse = []
k = range(1,10)

for i in k:
    kmeans_model = KMeans(n_clusters=i, random_state=42)
    kmeans_model.fit_predict(data_scaled)
    sse.append(kmeans_model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k, sse, 'bx-')
plt.xlabel("k")
plt.ylabel("SEE")
plt.title("Elbow Method for Optimal K")
plt.show()


#Agglomorative model Implementation
agglo_model = AgglomerativeClustering(n_clusters=3,linkage='complete')
agglo_res = agglo_model.fit_predict(data_scaled)
agglo_df = pd.DataFrame(agglo_res)

fig = plt.figure()
ax = fig.add_subplot(60)
scatter = ax.scatter(data["src_bytes"], data["dst_bytes"], c = agglo_df[0], s=50)
ax.set_title("Agglomorative Clustring")
ax.set_xlabel("src_bytes")
ax.set_ylabel("dst_bytes")
plt.colorbar(scatter)
plt.show()

#KNN for anomaly detection
knn_model = KNeighborsClassifier(n_neighbors=3) # NearestNeighbors(n_neighbors=3)
knn_pred = knn_model.fit_predict(data_scaled) # fit(data_scaled)


fig = plt.figure()
ax = fig.add_subplot(60)
scatter = ax.scatter(data["src_bytes"], data["dst_bytes"], c = knn_pred[0], s=50)
ax.set_title("Knn Clustring")
ax.set_xlabel("src_bytes")
ax.set_ylabel("dst_bytes")
plt.colorbar(scatter)
plt.show()