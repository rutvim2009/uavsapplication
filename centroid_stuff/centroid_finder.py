import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN


data = "/workspaces/uavsapplication/centroid_stuff/with outliers.txt"

with open(data, "r") as d:
    data_split = d.read().strip().split("\n")

N = int(data_split[0])
coords = np.array([list(map(float, line.split())) for line in data_split[1:]])


db = DBSCAN(eps=0.0005, min_samples=5).fit(coords) #start with 50 m and then reduce if necessary
mask = db.labels_ != -1
filtered = coords[mask]
if len(filtered) < 5:
    filtered = coords

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(filtered)
labels = kmeans.labels_

centroids = []
for i in range(5):
    cluster_points = filtered[labels == i]
    lat, lon = cluster_points.mean(axis=0) #using mean as centroid
    centroids.append((lat, lon))

centroids.sort(key=lambda x: x[0])

with open("/workspaces/uavsapplication/centroid_stuff/centers.out", "w") as f:
    for lat, lon in centroids:
        f.write(f"{lat:.5f} {lon:.5f}\n")

plt.figure(figsize=(8, 6))
plt.scatter(coords[:,1], coords[:,0], c="gray", s=10, alpha=0.5, label="all detections")
plt.scatter(filtered[:,1], filtered[:,0], c=labels, cmap="tab10", s=15, label="clusters")


for lat, lon in centroids:
    plt.scatter(lon, lat, c="red", s=120, marker="X", edgecolors="black", label="centroid")


plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("ODLC localization clusters")
plt.legend(loc="best")
plt.grid(True)
plt.show()
