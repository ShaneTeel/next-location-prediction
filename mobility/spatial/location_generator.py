import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score

from mobility.utils import centermost_point, get_logger

logger = get_logger(__name__)

class LocationGenerator:
    '''
    Purpose
    -------
    Perform clustering with DBSCAN for a single GeoLife User

    Actions
    -------
    - Cluster using lat/lon in Radians
    - Remove Outlier Clusters (-1)
    - Find the centroid (lat/lon) for each cluster
    - Merge cluster labels, stay points, and cluster centroids into a single cohesive data frame
    '''

    _KM_PER_RADIAN = 6371.0088

    def __init__(self, distance:float=0.2, min_k:int=1):
        '''
        Parameters
        ----------
        distance
        '''
        self.model = self._initialize_model(distance, min_k)
        self.score = -1

        logger.debug("Succesffully initialized SpatialEngineer.")

    def _initialize_model(self, distance, min_k):
        epsilon = distance / self._KM_PER_RADIAN
        logger.info(f"Initialized DBSCAN with epsilon == {epsilon:6f}")
        return DBSCAN(eps=epsilon, min_samples=min_k, metric='haversine', algorithm='ball_tree')

    def generate_locations(self, stay_points:pd.DataFrame):
        labels = self._cluster_staypoints(stay_points)
        self._evaluate(labels, stay_points)
        filtered = self._filter_noise(stay_points, labels)
        centroids = self._calculate_centroids(filtered)
        return self._merge_data(filtered, centroids)

    def _cluster_staypoints(self, stay_points:pd.DataFrame):
        coords = np.radians(stay_points[['lat', 'lon']])
        labels = self.model.fit(coords).labels_
        return labels

    def _evaluate(self, labels, stay_points:pd.DataFrame):
        coords = stay_points[["lat", "lon"]]
        self.score = davies_bouldin_score(coords, labels)
        logger.info(f"Clustering event resulted in a Davies Bouldin Index of {self.score:.4f}")

    def _filter_noise(self, stay_points:pd.DataFrame, labels):
        # Adding cluster labels back to dataframe
        clustered = pd.merge(
            left=stay_points,
            right=pd.Series(labels, name='cluster'),
            how='inner',
            left_index=True,
            right_index=True
        )

        # Filter Outliers/Noise (represented by -1)
        return clustered[clustered['cluster'] != -1]

    def _calculate_centroids(self, filtered_stay_points:pd.DataFrame):
        # Get Cluster Centroids by taking the median of all lat/lon for a specific cluster
        clusters = pd.Series({_: [coord for coord in coords.values] for _, coords in filtered_stay_points.groupby('cluster')[['lat', 'lon']]})
        return pd.DataFrame([[lat, lon] for lat, lon in clusters.map(centermost_point)], index=clusters.index).rename_axis('cluster').reset_index().rename(columns={0: 'cluster_lat', 1: 'cluster_lon'})
    
    def _merge_data(self, filtered:pd.DataFrame, centroids):
        merged = pd.merge(
               left=filtered, 
               right=centroids, 
               how='inner', 
               on='cluster'
               ).rename(columns={'lat': 'sp_lat', 'lon': 'sp_lon', "cluster": "origin_id"}).sort_values(by="arrived")
        
        return merged.loc[:, ["user_id", "origin_id", "arrived", "n_points", "sp_lat", "sp_lon", "cluster_lat", "cluster_lon", "departed", "duration"]] 

    def get_score(self):
        return self.score