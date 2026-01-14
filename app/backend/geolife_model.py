# backend/geolife_model.py
# Data Handling
import numpy as np
import pyproj
from datetime import datetime as dt, timedelta
import pandas as pd
pd.options.mode.copy_on_write = True
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import skmob
import glob
import os

# Data Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import HeatMap, HeatMapWithTime

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from skmob.preprocessing import filtering, detection, compression
from skmob.measures.individual import individual_mobility_network
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

# Modeling
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
import joblib as jb

# Evaluation
from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Helper Functions
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

# Load Data
def read_single_geolife_user(user_id:str):
    '''
    Purpose: Load a single GeoLife user's .plt files as a `skmob.TrajDataFrame`

    Inputs: The User's associated three-digit ID number, passed as a `str`

    Actions:
    1. Iterate through each .plt file for a single user and append all plt files as DataFrames to a DataFrame list
    2. Concatenate all DataFrames in list into a single DataFrame
    3. Convert Pandas DataFrame into a `skmob.TrajDataFrame`
    4. Return the `skmob.TrajDataFrame`
    '''
    data_dir = f'../data/geolife/{user_id}/Trajectory'
    single_users_daily_trajectories = []
    gps_traj = 1
    for single_user_file in glob.glob(os.path.join(data_dir, '*')):
        df = pd.read_csv(single_user_file, skiprows=6, header=None).rename({0: 'lat', 1:'lon', 3:'alt', 5:'date', 6: 'time'}, axis=1).drop(columns=[2, 4])
        df['utc'] = df['date'] + ' ' + df['time']
        df['utc'] = pd.to_datetime(df['utc'], utc=True, errors='coerce', format='%Y-%m-%d %H:%M:%S')
        df['beijing'] = df['utc'].dt.tz_convert('Asia/Shanghai')
        df['tid'] = gps_traj
        df['tid'] = df['tid'].ffill()
        df['uid'] = pd.Series(data_dir[16:19])
        df['uid'] = df['uid'].ffill()
        df['uid'] = df['uid'].astype('category')
        df.drop(columns=['date', 'time', 'utc', 'alt'], inplace=True)
        single_users_daily_trajectories.append(df)
        gps_traj += 1
    all_days_df = pd.concat(single_users_daily_trajectories, ignore_index=True)
    tdf = skmob.TrajDataFrame(
        data=all_days_df,
        latitude='lat',
        longitude='lon',
        datetime='beijing',
        trajectory_id='tid',
        user_id='uid',
        crs={'init': 'EPSG:4326'}
    )
    return tdf

# Clustering with DBSCAN
def cluster_geolife_user(uid, distance, min_k):
    '''
    Purpose: Perform clustering with DBSCAN for a single GeoLife User

    Inputs: A TrajDataFrame

    Actions:
    1. Preprocess
            - Noise Filtering
            - Trajectory Compression
            - Stop Detection
    2. DBSCAN Clustering
            - Cluster using lat/lng in Radians
            - Remove Outlier Clusters (-1)
            - Find the centroid (lat/lng) for each cluster
    3. Bringing it All Together
            - Merge labels, stop points, and centroids into a single cohesive TrajDataFrame
            - Calculate Scores using Davies-Bouldin Index and Calinski-Harabasz Score
    4. Return 
            - A TrajDataFrame with cluster labels, centroids, and stop points
            - A dictionary with the scores
    '''   
    # Create copy
    tdf = read_single_geolife_user(uid)
    
    # # Preprocess for Clustering
    # Remove Noise
    filtered = filtering.filter(tdf, max_speed_kmh=250).sort_by_uid_and_datetime()

    # Compress points
    compressed = compression.compress(filtered, spatial_radius_km=.2).sort_by_uid_and_datetime()

    # Identify Stop Locations
    detected = detection.stay_locations(compressed, minutes_for_a_stop=60).sort_by_uid_and_datetime().reset_index(drop=True)
    # Cluster Inputs (in radians)
    coords = np.radians(detected[['lat', 'lng']])
    kms_per_radian = 6371.0088
    distance = distance
    epsilon = distance / kms_per_radian
    min_k = min_k   

    # Cluster Object (returning labels only)
    labels = DBSCAN(eps=epsilon, min_samples=min_k, metric='haversine', algorithm='ball_tree').fit(coords).labels_ 

    # Adding cluster labels back to dataframe
    clustered = pd.merge(
        left=detected,
        right=pd.Series(labels, name='cluster'),
        how='inner',
        left_index=True,
        right_index=True
    )

    # Filter Outliers/Noise (represented by -1)
    clusters_filtered = clustered[clustered['cluster'] != -1]

    # Get Cluster Centroids by taking the median of all lat/lng for a specific cluster
    clusters = pd.Series({_: [coord for coord in coords.values] for _, coords in clusters_filtered.groupby('cluster')[['lat', 'lng']]})
    cluster_centroids = pd.DataFrame([[lat, lng] for lat, lng in clusters.map(get_centermost_point)], index=clusters.index).rename_axis('cluster').reset_index().rename(columns={0: 'cluster_lat', 1: 'cluster_lng'})

    # Merge cluster_centroids with cluster labels
    final_merge = pd.merge(
            left=clusters_filtered[['datetime', 'uid', 'tid', 'cluster', 'leaving_datetime', 'lat', 'lng']], 
            right=cluster_centroids, 
            how='inner', 
            on='cluster', 
            suffixes=(None, '_cluster')).rename(columns={'lat': 'precise_lat', 'lng': 'precise_lng'}).sort_values(by='datetime')
    
    cluster_location_key = pd.DataFrame(final_merge).iloc[:, [1, 0, 5, 6, 3, 7, 8, 4]]

    # Evaluate
    DBI = davies_bouldin_score(X=clusters_filtered[['lat', 'lng']], labels=clusters_filtered['cluster'])
    CHI = calinski_harabasz_score(X=clusters_filtered[['lat', 'lng']], labels=clusters_filtered['cluster'])
    scores = {
        'Davies-Bouldin Index': DBI,
        'Calinski-Harabasz Score': CHI
    }    
    # Ceate Mobility Network of User specific mobility
    mobility_network = individual_mobility_network(
        skmob.TrajDataFrame(
            data=cluster_location_key,
            latitude='precise_lat',
            longitude='precise_lng',
            datetime='datetime',
            user_id='uid',
            trajectory_id='tid'
        ),
        self_loops=True
    )

    # Merge with cluster location key to obtain cluster location of origin
    origin_with_clusters = pd.merge(
        left=mobility_network.drop(columns=['n_trips', 'uid']), 
        right=cluster_location_key[['datetime', 'leaving_datetime', 'cluster', 'cluster_lat', 'cluster_lng', 'precise_lat', 'precise_lng', 'uid']], 
        how='inner', 
        left_on=['lat_origin', 'lng_origin'], 
        right_on=['precise_lat', 'precise_lng']
    ).drop(columns=['precise_lat', 'precise_lng']).rename(columns={'cluster': 'cluster_origin', 'cluster_lat': 'cluster_origin_lat', 'cluster_lng': 'cluster_origin_lng'})

    # Merge with cluster location key to obtain cluster location of destination
    ml_df = pd.merge(
        left=origin_with_clusters, 
        right=cluster_location_key[['cluster', 'precise_lat', 'cluster_lat', 'cluster_lng', 'precise_lng']], 
        how='inner', 
        left_on=['lat_dest', 'lng_dest'], 
        right_on=['precise_lat', 'precise_lng']
    ).rename(columns={'cluster': 'cluster_dest', 'cluster_lat': 'cluster_dest_lat', 'cluster_lng': 'cluster_dest_lng', 'precise_lat': 'dest_lat', 'precise_lng': 'dest_lng'})

    # Create timeseries features
    ml_df['uid'] = ml_df['uid']
    ml_df['datetime'] = pd.to_datetime(ml_df['datetime'])
    ml_df['leaving_datetime'] = pd.to_datetime(ml_df['leaving_datetime'])
    ml_df['month'] = ml_df['datetime'].dt.month
    ml_df['day_of_week'] = ml_df['datetime'].dt.day_of_week
    ml_df['day'] = ml_df['datetime'].dt.day
    ml_df['hour_in_day'] = ml_df['datetime'].dt.hour
    ml_df['minute_in_hour'] = ml_df['datetime'].dt.minute
    ml_df['timedelta'] = (ml_df['leaving_datetime'] - ml_df['datetime']) / timedelta(minutes=1)
    ml_df.sort_values(by='datetime', inplace=True)
    ml_df.drop(columns='leaving_datetime', inplace=True)
    ml_df = ml_df.loc[:, ['uid', 'lat_origin', 'lng_origin', 'cluster_origin', 'cluster_origin_lat', 'cluster_origin_lng', 'datetime', 'month', 'day', 'day_of_week', 'hour_in_day', 'minute_in_hour', 'timedelta', 'dest_lat', 'dest_lng', 'cluster_dest', 'cluster_dest_lat', 'cluster_dest_lng']]
    return ml_df, scores

def model_pipe(ml_df, thresh:int=6, min_samples:int=5):
    '''
    Purpose: Generate a next-location prediction model for a single user in the GeoLife dataset

    Parameters: The DataFrame returned by the `feature_engineer()` function
    
    Actions:
    1. Create Variables `X` and `y`
    2. Create Training & Test Splits (for evaluating model performance)
    3. Instatiate Objects for modeling
            - `MinMaxScaler()` for numeric inputs
            - `OneHotEncoder()` for categorical inputs
            - `ColumnTransformer()` to handle preprocessing
            - `SMOTENC()` to oversample minority classes
            - `RandomForestClassifier()` as the primary estimator
            - `OnevsOneClassifer()` as the strategy for handling the classes
            - `Pipeline()` for bringing everything together
    4. Fit/Train, Predict, and Score model
    5. Create Metadata and assing to model
    6. Return fitted model
    '''

    origin_mask = ml_df.loc[:, 'cluster_origin'].value_counts()
    dest_mask = ml_df.loc[:, 'cluster_dest'].value_counts()

    final_ml_df = ml_df[(ml_df.loc[:, 'cluster_origin'].isin(origin_mask[origin_mask.values > thresh].index)) & (ml_df.loc[:, 'cluster_dest'].isin(dest_mask[dest_mask.values > thresh].index))]

    # Create Variables
    X = final_ml_df.drop(columns='cluster_dest').reset_index(drop=True)
    y = final_ml_df['cluster_dest'].reset_index(drop=True)

    # Create Training & Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

    # Instatiate Preprocessor Objects
    scaler = MinMaxScaler()
    ohe = OneHotEncoder(sparse_output=False, drop='first', dtype=int, handle_unknown='ignore')

    # Define Features for Preprocessing by type
    categorical_features = ['uid', 'month', 'day_of_week']
    numeric_features = ['lat_origin', 'lng_origin', 'day', 'hour_in_day', 'minute_in_hour', 'timedelta']

    # Instatiate Column Transformer Object
    # Preprocessor Object
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', ohe, categorical_features),
            ('num', scaler, numeric_features),
        ],
            remainder='drop',
            force_int_remainder_cols=False
    )

    # Instatiate Over-Sampler
    sampler = SMOTENC(
        k_neighbors=min_samples, 
        categorical_features=[n for n in range(0, 15)],  
        random_state=42, 
        sampling_strategy='all'
    )

    # Instatiate Classifier Objects
    classifier = RandomForestClassifier(random_state=42, bootstrap=True, oob_score=True, class_weight='balanced_subsample')
    strategizer = OneVsOneClassifier(classifier)

    # Bring it All Together with Instatiated Model Pipeline
    model_pipe = Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('sampler', sampler),
            ('classifier', strategizer)
        ]
    )

    # Fit/Train Model
    model_pipe.fit(X_train, y_train)

    # Feature Importance for Metadata
    feature_importances = {name: stat for name, stat in zip(preprocessor.get_feature_names_out(), np.mean([arr for arr in [estimator.feature_importances_ for estimator in strategizer.estimators_]], axis=0))}

    # Get Predictions and Decision Calculus
    predictions = model_pipe.predict(X_test)
    y_score = model_pipe.decision_function(X_test)

    # Evaluate Model
    f1 = f1_score(y_test, predictions, zero_division=0, average='weighted')
    top_k_accuracy = top_k_accuracy_score(y_test, y_score)
    AP_AUC = average_precision_score(y_test, y_score, average='weighted')
    cohen_kappa = cohen_kappa_score(y_test, predictions)

    # Create Metadata For Model
    scores = {
            'F1 (Weighted)': f1,
            'Top K Acc. (Top 2)': top_k_accuracy,
            'Avg. Precision AUC': AP_AUC,
            "Cohen's Kappa Score": cohen_kappa
    }
    return scores, final_ml_df

def save_model(ml_df, uid, thresh:int=6, min_samples:int=5):
    '''
    Purpose: Generate a next-location prediction model for a single user in the GeoLife dataset

    Parameters: The DataFrame returned by the `feature_engineer()` function
    
    Actions:
    1. Create Variables `X` and `y`
    2. Create Training & Test Splits (for evaluating model performance)
    3. Instatiate Objects for modeling
            - `MinMaxScaler()` for numeric inputs
            - `OneHotEncoder()` for categorical inputs
            - `ColumnTransformer()` to handle preprocessing
            - `SMOTENC()` to oversample minority classes
            - `RandomForestClassifier()` as the primary estimator
            - `OnevsOneClassifer()` as the strategy for handling the classes
            - `Pipeline()` for bringing everything together
    4. Fit/Train, Predict, and Score model
    5. Create Metadata and assing to model
    6. Return fitted model
    '''

    origin_mask = ml_df.loc[:, 'cluster_origin'].value_counts()
    dest_mask = ml_df.loc[:, 'cluster_dest'].value_counts()

    final_ml_df = ml_df[(ml_df.loc[:, 'cluster_origin'].isin(origin_mask[origin_mask.values > thresh].index)) & (ml_df.loc[:, 'cluster_dest'].isin(dest_mask[dest_mask.values > thresh].index))]

    # Create Variables
    X = final_ml_df.drop(columns='cluster_dest').reset_index(drop=True)
    y = final_ml_df['cluster_dest'].reset_index(drop=True)

    # Instatiate Preprocessor Objects
    scaler = MinMaxScaler()
    ohe = OneHotEncoder(sparse_output=False, drop='first', dtype=int, handle_unknown='ignore')

    # Define Features for Preprocessing by type
    categorical_features = ['uid', 'month', 'day_of_week']
    numeric_features = ['lat_origin', 'lng_origin', 'day', 'hour_in_day', 'minute_in_hour', 'timedelta']

    # Instatiate Column Transformer Object
    # Preprocessor Object
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', ohe, categorical_features),
            ('num', scaler, numeric_features),
        ],
            remainder='drop',
            force_int_remainder_cols=False
    )

    # Instatiate Over-Sampler
    sampler = SMOTENC(
        k_neighbors=min_samples, 
        categorical_features=[n for n in range(0, 15)],  
        random_state=42, 
        sampling_strategy='all'
    )

    # Instatiate Classifier Objects
    classifier = RandomForestClassifier(random_state=42, bootstrap=True, oob_score=True, class_weight='balanced_subsample')
    strategizer = OneVsOneClassifier(classifier)

    # Bring it All Together with Instatiated Model Pipeline
    model_pipe = Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('sampler', sampler),
            ('classifier', strategizer)
        ]
    )

    # Fit/Train Model
    model_pipe.fit(X, y)
    uid = int(uid) 
    # Feature Importance for Metadata
    feature_importances = {name: stat for name, stat in zip(preprocessor.get_feature_names_out(), np.mean([arr for arr in [estimator.feature_importances_ for estimator in strategizer.estimators_]], axis=0))}
    file_path = f'../model/geolife_nlp_user{uid}.pkl'
    jb.dump(model_pipe, file_path) 
    final_ml_df.to_csv(f'../model/geolife_clusters_user{uid}.csv', index=False)
    return final_ml_df

def load_geolife_model(uid):
    model_path = f"../model/geolife_nlp_user{uid}.pkl"
    df = pd.read_csv(f'../model/geolife_clusters_user{uid}.csv')
    
    model = jb.load(model_path)
    
    feature_names = ['uid', 'lat_origin', 'lng_origin', 'timedelta', 'month', 'day', 
                     'day_of_week', 'hour_in_day', 'minute_in_hour'
                    ]
    
    return model, feature_names, df

if __name__ == "__main__":
    #train_model()
    # model, feature_names = load_car_model()
    # input_data = pd.DataFrame([[4, 200.0, 150.0, 3000.0, 15.0, 76, 1]], 
    #                          columns=feature_names)
    
    #input_data = np.array([[4, 200.0, 150.0, 3000.0, 15.0, 76, 1]])
    # print(model.predict(input_data))
    pass