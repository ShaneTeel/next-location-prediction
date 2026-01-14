import pandas as pd
import numpy as np

from mobility.utils import get_logger

logger = get_logger(__name__)

class DataQualityAssessment:
    """
    Evaluates collection quality to inform confidence in assessments.
    Implements ICD 203 Analytic Standard: Sourcing
    """
    
    def __init__(self, user_id: str, positionfixes: pd.DataFrame):
        self.user_id = user_id
        self.data = positionfixes
        self.temporal_metrics = self._assess_temporal_coverage()
        self.collection_metrics = self._assess_collection_density()
        self.gaps = self._identify_gaps()
        self.overall_reliability = self._calculate_reliability()
        self.quality_metrics = {
            "temporal_coverage": self.temporal_metrics,
            "collection_density": self.collection_metrics,
            "gaps": self.gaps,
            "overall_reliability": self.overall_reliability
        }
        
        logger.debug(f"Source quality assessment initialized for {self.user_id}")
    
    def _assess_temporal_coverage(self):
        """
        How well does collection cover the time period?
        """
        date_range = (self.data['datetime'].max() - 
                     self.data['datetime'].min()).days
        
        active_days = self.data['datetime'].dt.date.nunique()
        coverage_ratio = active_days / date_range if date_range > 0 else 0
        
        return {
            'total_days': date_range,
            'active_days': active_days,
            'coverage_ratio': coverage_ratio,
            'assessment': self._interpret_coverage(coverage_ratio)
        }
    
    def _interpret_coverage(self, coverage_ratio):
        """Translate coverage ratio to confidence assessment"""
        if coverage_ratio >= 0.8:
            return "HIGH confidence: Comprehensive temporal coverage"
        elif coverage_ratio >= 0.5:
            return "MODERATE confidence: Adequate coverage with gaps"
        else:
            return "LOW confidence: Sparse coverage limits pattern detection"
    
    def _assess_collection_density(self):
        """
        How frequently are we collecting data?
        """
        time_diffs = self.data['datetime'].diff().dt.total_seconds() / 60
        median_gap = time_diffs.median()
        
        return {
            'median_gap_minutes': median_gap,
            'assessment': self._interpret_density(median_gap)
        }
    
    def _interpret_density(self, median_gap):
        """Assess collection frequency adequacy"""
        if median_gap <= 5:
            return "HIGH confidence: Very frequent collection"
        elif median_gap <= 30:
            return "HIGH confidence: Frequent collection"
        elif median_gap <= 120:
            return "MODERATE confidence: Adequate for routine analysis"
        else:
            return "LOW confidence: Sparse collection may miss activities"
    
    def _identify_gaps(self):
        """
        Flag significant collection gaps (>24 hours).
        """
        gaps = []
        sorted_data = self.data.sort_values('datetime')
        time_diffs = sorted_data['datetime'].diff()
        gap_threshold = pd.Timedelta(hours=24)
        large_gaps = time_diffs[time_diffs > gap_threshold]
        
        for idx in large_gaps.index:
            prev_idx = sorted_data.index.get_loc(idx) - 1
            if prev_idx >= 0:
                prev_time = sorted_data.iloc[prev_idx]['datetime']
                curr_time = sorted_data.loc[idx, 'datetime']
                
                gaps.append({
                    'start': prev_time,
                    'end': curr_time,
                    'duration_hours': (curr_time - prev_time).total_seconds() / 3600
                })
        
        return gaps
    
    def _calculate_reliability(self):
        """
        Overall source reliability score (0-1).
        """
        temporal = self.temporal_metrics['coverage_ratio']
        num_gaps = len(self.gaps)
        gap_penalty = min(0.3, num_gaps * 0.02)
        
        median_gap = self.collection_metrics['median_gap_minutes']
        if median_gap <= 30:
            density_score = 1.0
        elif median_gap <= 120:
            density_score = 0.8
        else:
            density_score = 0.6
        
        reliability = max(0, min(1, temporal * density_score - gap_penalty))
        return round(reliability, 2)
    
    def generate_source_statement(self):
        """
        Generate ICD 203-compliant source description.
        """
        metrics = self.quality_metrics
        
        statement = f"""
SOURCE ASSESSMENT (ICD 203 Analytic Standard: Sourcing)

User: {self.user_id}

Source Description:
GPS trajectory data collected over {metrics['temporal_coverage']['total_days']} days 
with {metrics['temporal_coverage']['active_days']} days of active collection 
({metrics['temporal_coverage']['coverage_ratio']:.0%} temporal coverage).

Source Reliability: {metrics['overall_reliability']:.2f}/1.00

Temporal Coverage: {metrics['temporal_coverage']['assessment']}

Collection Density: 
Median gap: {metrics['collection_density']['median_gap_minutes']:.1f} minutes
{metrics['collection_density']['assessment']}

Identified Gaps: {len(metrics['gaps'])} significant collection gaps (>24 hours)
"""
        
        if metrics['gaps']:
            statement += "\nNotable gaps:\n"
            for gap in sorted(metrics['gaps'], key=lambda x: x['duration_hours'], reverse=True)[:5]:
                statement += f"  - {gap['start'].date()} to {gap['end'].date()} ({gap['duration_hours']:.0f} hours)\n"
        
        reliability = metrics['overall_reliability']
        confidence_level = "HIGH" if reliability >= 0.8 else "MODERATE" if reliability >= 0.5 else "LOW"
        
        statement += f"""
Confidence Implications:
- Pattern-of-life assessments: {confidence_level} confidence
- Anomaly detection: {"HIGH" if len(metrics['gaps']) < 5 else "MODERATE" if len(metrics['gaps']) < 15 else "LOW"} confidence
- Predictive assessments: Qualified by temporal gaps and collection density

Assumptions:
- Lack of collection does not equal lack of activity
- Gaps in data may obscure deviations from routine
- Confidence in routine assessment increases with collection duration
"""
        
        return statement
    
    def get_summary(self):
        """Return summary dict for dashboard"""
        return {
            'user_id': self.user_id,
            'reliability_score': self.quality_metrics['overall_reliability'],
            'total_days': self.quality_metrics['temporal_coverage']['total_days'],
            'active_days': self.quality_metrics['temporal_coverage']['active_days'],
            'coverage_ratio': self.quality_metrics['temporal_coverage']['coverage_ratio'],
            'median_gap_minutes': self.quality_metrics['collection_density']['median_gap_minutes'],
            'num_gaps': len(self.quality_metrics['gaps']),
            'major_gaps': sorted(self.quality_metrics['gaps'], 
                                key=lambda x: x['duration_hours'], 
                                reverse=True)[:5]
        }