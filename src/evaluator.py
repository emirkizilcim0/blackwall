import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, accuracy_score)
import pandas as pd
import numpy as np

class CyberEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model_name, model, X_test, y_test, y_pred=None):
        """Comprehensive model evaluation"""
        # If y_pred not provided, predict using model
        if y_pred is None:
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                raise ValueError("Model must have predict method or provide y_pred")
        
        # For IsolationForest, convert predictions
        if model_name == 'IsolationForest':
            y_pred = (y_pred == -1).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Handle case where only one class is predicted
        if len(np.unique(y_pred)) == 1:
            precision = 1.0 if np.unique(y_pred)[0] == 1 else 0.0
            recall = 1.0 if np.unique(y_pred)[0] == 1 else 0.0
            f1 = 1.0 if np.unique(y_pred)[0] == 1 else 0.0
        else:
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
        
        # ROC AUC if probabilities available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            self.results[model_name]['roc_auc'] = roc_auc_score(y_test, y_prob)
            self.results[model_name]['y_prob'] = y_prob
        
        return self.results[model_name]
    
    def plot_performance_dashboard(self):
        """Create a comprehensive performance dashboard"""
        if not self.results:
            print("⚠️  No results to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🛡️ BlackWall Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Model Comparison Bar Chart
        metrics_df = pd.DataFrame(self.results).T
        if all(metric in metrics_df.columns for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
            metrics_df[['accuracy', 'precision', 'recall', 'f1_score']].plot(
                kind='bar', ax=axes[0,0], title='Model Performance Comparison'
            )
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Confusion Matrix for Best Model
        if self.results:
            best_model = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_score'])
            cm = self.results[best_model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r', ax=axes[0,1])
            axes[0,1].set_title(f'Confusion Matrix - {best_model}')
            axes[0,1].set_ylabel('Actual')
            axes[0,1].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('blackwall_performance.png', dpi=300, bbox_inches='tight')
        plt.show()