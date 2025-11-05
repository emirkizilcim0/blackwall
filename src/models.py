from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import numpy as np

class BlackWallModels:
    def __init__(self):
        self.models = {}
        self.best_params = {}
    
    def initialize_models(self):
        """Initialize all ML models for BlackWall"""
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            # 'SVM': SVC(random_state=42, probability=False),             # Checking the probability flag
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'IsolationForest': IsolationForest(random_state=42, contamination=0.1)
        }
        return self.models
    
    def hyperparameter_tuning(self, model_name, model, X_train, y_train):
        """Perform hyperparameter tuning for selected models"""
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if model_name in param_grids:
            print(f"🎯 Tuning {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], 
                                    cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.best_params[model_name] = grid_search.best_params_
            return grid_search.best_estimator_
        
        return model