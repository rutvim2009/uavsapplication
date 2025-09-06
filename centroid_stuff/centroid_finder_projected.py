# same code as centroid_finder.py but converts the coordinates to meters using UTM

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans, DBSCAN


data = "/workspaces/uavsapplication/with outliers.txt"  

with open(data, "r") as f:
    lines = f.read().strip().split("\n")

N = int(lines[0])
coords = np.array([list(map(float, line.split())) for line in lines[1:]])


gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(coords[:,1], coords[:,0]),  
    crs="EPSG:4326"
)


mean_lon = coords[:,1].mean()
utm_zone = int((mean_lon + 180) / 6) + 1
epsg_code = 32600 + utm_zone  

gdf = gdf.to_crs(epsg=epsg_code)

proj_coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T


db = DBSCAN(eps=15, min_samples=5).fit(proj_coords)  
mask = db.labels_ != -1
filtered = proj_coords[mask]

if len(filtered) < 5:
    filtered = proj_coords

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(filtered)
labels = kmeans.labels_


centroids_proj = []
for i in range(5):
    cluster_points = filtered[labels == i]
    x, y = cluster_points.mean(axis=0)
    centroids_proj.append((x, y))


centroids_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy([c[0] for c in centroids_proj],
                                [c[1] for c in centroids_proj]),
    crs=f"EPSG:{epsg_code}"
)
centroids_wgs84 = centroids_gdf.to_crs(epsg=4326)

centroids = [(pt.y, pt.x) for pt in centroids_wgs84.geometry]
centroids.sort(key=lambda x: x[0])


for lat, lon in centroids:
    print(f"{lat:.5f} {lon:.5f}")
