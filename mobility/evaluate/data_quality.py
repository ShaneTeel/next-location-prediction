import numpy as np
import pandas as pd
from datetime import timedelta

from mobility.utils import get_logger

class DataQuality:

    def __init__(self, user_id:str, pfs:pd.DataFrame):
        self.user_id:str = user_id
        self.data:pd.DataFrame = pfs
        
    def _assess_temporal_coverage(self):
        '''
        How well does the data coverage period account of each day of potential activity
        '''
        latest = self.data["datetime"].max()
        earliest = self.data["datetime"].min()
        total_days = (latest - earliest).days

        active_days = self.data["datetime"].dt.date.nunique()
        coverage = active_days / total_days if total_days > 0 else 0
        return {
            "Total Days": total_days,
            "Active Days": active_days,
            "Coverage Ratio": coverage
        }
    
    def _assess_collection_density(self):
        density = {}
        
        self.data["day_of_week"] = self.data["datetime"].dt.day_of_week
        days_of_week = self.data["day_of_week"].unique()
        for day in days_of_week:
            df = self.data[self.data["day_of_week"] == day]
            time_diff = df["datetime"].diff().dt.total_seconds() / 60
            density[day] = time_diff.median()
    
        return density

    def _assess_gaps(self):
        gaps = []
        sorted_data = self.data.sort_values(by="datetime")
        
        time_delta = timedelta(hours=24)
        