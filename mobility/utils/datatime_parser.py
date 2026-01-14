import numpy as np
import pandas as pd

def datetime_parser(self, df:pd.DataFrame):
    df = df.copy() 
    df['month'] = df["arrived"].dt.month.astype('category')
    df['day_of_week'] = df["arrived"].dt.day_of_week.astype('category')
    df['day'] = df["arrived"].dt.day
    df['arrival_hour'] = df["arrived"].dt.hour
    df['arrival_minute'] = df["arrived"].dt.minute
    conditions = [
        (df['arrival_hour'] >= 5) & (df['arrival_hour'] < 12),
        (df['arrival_hour'] >= 12) & (df['arrival_hour'] < 17),
        (df['arrival_hour'] >= 17) & (df['arrival_hour'] < 22)
    ]
    options = ['morning', 'afternoon', 'evening']
    df['arrival_time_of_day'] = np.select(conditions, options, default='night')

    df["departure_hour"] = df["departed"].dt.hour
    df["departure_minute"] = df["departed"].dt.minute

    if self.is_weekend:
        df['is_weekend'] = [1 if day >= 5 else 0 for day in df["day_of_week"]].astype('category')
    df.sort_values(by="datetime", inplace=True)
    df.drop(columns=["arrived", "departed"], inplace=True)
    return df