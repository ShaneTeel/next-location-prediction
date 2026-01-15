import numpy as np
import pandas as pd

def datetime_parser(df:pd.DataFrame):
    df = df.copy() 
    df['month'] = df["arrived"].dt.month.astype('category')
    df['day_of_week'] = df["arrived"].dt.day_of_week.astype('category')
    df['day'] = df["arrived"].dt.day
    df['hour'] = df["arrived"].dt.hour
    df['minute'] = df["arrived"].dt.minute
    conditions = [
        (df['hour'] >= 5) & (df['hour'] < 12),
        (df['hour'] >= 12) & (df['hour'] < 17),
        (df['hour'] >= 17) & (df['hour'] < 22)
    ]
    options = ['morning', 'afternoon', 'evening']
    df['time_of_day'] = np.select(conditions, options, default='night')

    df['is_weekend'] = df["day_of_week"].transform(lambda day: 1 if day >= 5 else 0).astype('category')
    df.sort_values(by="arrived", inplace=True)
    df.drop(columns=["arrived"], inplace=True)
    return df