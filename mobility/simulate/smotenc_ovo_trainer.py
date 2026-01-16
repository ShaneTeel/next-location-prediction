import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
    average_precision_score,
    cohen_kappa_score
)

from typing import Literal

class SMOTENC_OvO_Trainer:

    def __init__(self, cat_features:list, num_features:list, k_neighbors:int=5, sampling_strategy:str="auto", bootstrap:bool=True, oob_score:bool=True, class_weight:str="balanced_subsample"):
        self.cat_features = cat_features
        self.num_features = num_features
        self.preprocessor = self._initialize_preprocessor(cat_features=cat_features, num_features=num_features)
        self.sampler = self._initialize_sampler(k_neighbors, sampling_strategy)
        self.classifier = self._intialize_classifier(bootstrap, oob_score, class_weight)
        self.pipeline = self._initialize_pipeline(
            self.preprocessor,
            self.sampler,
            self.classifier
        )

    def _initialize_preprocessor(self, cat_features, num_features):
        encoder = OneHotEncoder(sparse_output=False, drop="first", dtype=int, handle_unknown="ignore")
        scaler = MinMaxScaler()
        return ColumnTransformer(
            transformers=[
                ("cat", encoder, cat_features),
                ("num", scaler, num_features)
            ],
            remainder="drop"
        )
    
    def _initialize_sampler(self, k_neighbors:int=5, sampling_strategy:str="auto"):
        return SMOTENC(
            categorical_features=list(range(0, 15)),
            k_neighbors=k_neighbors,
            random_state=42,
            sampling_strategy=sampling_strategy
        )
    
    def _intialize_classifier(self, bootstrap:bool=True, oob_score:bool=True, class_weight:Literal["balanced_subsample",]="balanced_subsample"):
        estimator = RandomForestClassifier(random_state=42, bootstrap=bootstrap, oob_score=oob_score, class_weight=class_weight)
        return OneVsOneClassifier(estimator=estimator)
    
    def _initialize_pipeline(self, preprocessor, sampler, classifier):
        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("sampler", sampler),
                ("classifier", classifier)
            ]
        )

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        avg_feature_importance = np.mean([arr for arr in [estimator.feature_importances_ for estimator in self.classifier.estimators_]], axis=0)
        self.feature_importance_dict = {name: stat for name, stat in zip(self.preprocessor.get_feature_names_out(), avg_feature_importance)}
        return self.pipeline
    
    def test(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        y_score = self.pipeline.decision_function(X_test)
        return y_pred, y_score
    
    def evaluate(self, y_true, y_pred, y_score):
        f1 = f1_score(y_true, y_pred, zero_division=0, average="weighted")
        top_k = top_k_accuracy_score(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score, average="weighted")
        cohen_kappa = cohen_kappa_score(y_true, y_pred)

        return {
            "F1 (Weighted)": f1,
            "Top K Accuracy": top_k,
            "Avg. Precision": avg_prec,
            "Cohen's Kappa": cohen_kappa
        }
