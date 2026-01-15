import pandas as pd
from sklearn.model_selection import train_test_split

def input_generator(df:pd.DataFrame):
    origin_mask = df.loc[:, 'origin_id'].value_counts()
    dest_mask = df.loc[:, 'dest_id'].value_counts()

    df = df[(df.loc[:, 'origin_id'].isin(origin_mask[origin_mask.values > 4].index)) & (df.loc[:, 'dest_id'].isin(dest_mask[dest_mask.values > 4].index))]
    

    X = df.drop(columns="dest_id")
    y = df[["dest_id"]]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
