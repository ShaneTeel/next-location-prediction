import numpy as np
import pandas as pd

from typing import Literal

from .markov_chain import MarkovChain
from mobility.utils import get_logger

logger = get_logger(__name__)

class TimeAwareMarkov:
    _SEGMENTs = ["hour", "time_of_day", "day_of_week"]

    def __init__(self, data:pd.DataFrame, datetime_col_name:str, states_col_name:str):
       self.data = self._parse_datetime(data, datetime_col_name)
       self.states = self.data[states_col_name]
       self.models = self._initialize_models()

       logger.debug("TimeAwareMarkov successfully initialized.")
    
    def _initialize_models(self, states:pd.Series, data:pd.DataFrame):
        models = {
            "hours": {}
        }
        hours = self.data["hours"].unique()
        time_of_day = self.data["time_of_day"].unique()
        day_of_week = self.data["day_of_week"].unique()


    def _parse_datetime(df:pd.DataFrame, col_name:str):
        df = df.copy() 
        df['day_of_week'] = df[col_name].dt.day_of_week.astype('category')
        df['hour'] = df[col_name].dt.hour
        conditions = [
            (df['hour'] >= 5) & (df['hour'] < 12),
            (df['hour'] >= 12) & (df['hour'] < 17),
            (df['hour'] >= 17) & (df['hour'] < 22)
        ]
        options = ['morning', 'afternoon', 'evening']
        df['time_of_day'] = np.select(conditions, options, default='night')

        df.sort_values(by=col_name, inplace=True)
        df.drop(columns=[col_name], inplace=True)
        return df