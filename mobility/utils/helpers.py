import numpy as np
import pandas as pd
from datetime import timedelta
import pyproj


def datetime_creator(df, datetime='datetime', departure_time='leaving_datetime'):
    df = df.copy() 
    df['uid'] = df['uid'].astype('category')
    df[datetime] = pd.to_datetime(df[datetime])
    df[departure_time] = pd.to_datetime(df[departure_time])
    df['month'] = df[datetime].dt.month.astype('category')
    df['day_of_week'] = df[datetime].dt.day_of_week.astype('category')
    df['day'] = df[datetime].dt.day
    df['hour_in_day'] = df[datetime].dt.hour
    df['minute_in_hour'] = df[datetime].dt.minute
    conditions = [
        (df['hour_in_day'] >= 5) & (df['hour_in_day'] < 12),
        (df['hour_in_day'] >= 12) & (df['hour_in_day'] < 17),
        (df['hour_in_day'] >= 17) & (df['hour_in_day'] < 22)
    ]
    options = ['morning', 'afternoon', 'evening']
    df['time_of_day'] = np.select(conditions, options, default='night')
    df['is_weekend'] = [1 if day >= 5 else 0 for day in df[datetime].dt.day_of_week].astype('category')
    df['timedelta'] = (df[departure_time] - df[datetime]) / timedelta(minutes=1)
    df.sort_values(by=datetime, inplace=True)
    df.drop(columns=[datetime, departure_time], inplace=True)
    return df

def lla_to_ecf(lat, lng, alt):
    transformer = pyproj.Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
    x, y, z = transformer.transform(lng, lat, alt)
    df = pd.DataFrame(data=zip(x, y, z))
    return df