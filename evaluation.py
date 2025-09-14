"""
Evaluation and testing framework for Valorant ML model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, learning_curve
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ValorantModelEvaluator:
    """Comprehensive evaluation framework for Valorant prediction models"""
    
    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive model performance evaluation"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': self._calculate_auc_pr(y_test, y_pred_proba)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def _calculate_auc_pr(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Area Under Precision-Recall Curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return np.trapz(precision, recall)
    
    def plot_roc_curve(self, y_test: np.ndarray, y_pred_proba: np.ndarray, 
                      title: str = "ROC Curve"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_test: np.ndarray, y_pred_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve"):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {auc_pr:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix_detailed(self, y_test: np.ndarray, y_pred: np.ndarray,
                                     title: str = "Confusion Matrix"):
        """Plot detailed confusion matrix with percentages"""
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Team 2 Wins', 'Team 1 Wins'],
                   yticklabels=['Team 2 Wins', 'Team 1 Wins'])
        ax1.set_title(f'{title} - Counts')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                   xticklabels=['Team 2 Wins', 'Team 1 Wins'],
                   yticklabels=['Team 2 Wins', 'Team 1 Wins'])
        ax2.set_title(f'{title} - Percentages')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_confidence(self, y_test: np.ndarray, y_pred_proba: np.ndarray,
                                 title: str = "Prediction Confidence Distribution"):
        """Plot distribution of prediction confidence"""
        correct_predictions = (y_test == (y_pred_proba > 0.5).astype(int))
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[correct_predictions], bins=20, alpha=0.7, 
                label='Correct Predictions', color='green')
        plt.hist(y_pred_proba[~correct_predictions], bins=20, alpha=0.7, 
                label='Incorrect Predictions', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Confidence vs Accuracy
        plt.subplot(1, 2, 2)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        
        for i in range(len(bins) - 1):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i + 1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Accuracy')
        plt.title('Confidence vs Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curve(self, X: np.ndarray, y: np.ndarray, 
                          title: str = "Learning Curve"):
        """Plot learning curve to check for overfitting/underfitting"""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, top_n: int = 20):
        """Analyze and plot feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return None
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(top_n), 
                   x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def cross_validation_analysis(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Perform cross-validation analysis"""
        print(f"Performing {cv}-fold cross-validation...")
        
        # Different scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            
            print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Plot CV results
        plt.figure(figsize=(12, 6))
        
        metrics = list(cv_results.keys())
        means = [cv_results[metric]['mean'] for metric in metrics]
        stds = [cv_results[metric]['std'] for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{cv}-Fold Cross-Validation Results')
        plt.xticks(x_pos, [m.upper() for m in metrics], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cv_results
    
    def generate_evaluation_report(self, X_test: np.ndarray, y_test: np.ndarray, 
                                 X_train: np.ndarray = None, y_train: np.ndarray = None) -> str:
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")
        
        # Basic performance metrics
        metrics, y_pred, y_pred_proba = self.evaluate_performance(X_test, y_test)
        
        report = f"""
VALORANT MATCH PREDICTION MODEL EVALUATION REPORT
================================================

MODEL PERFORMANCE METRICS:
--------------------------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1_score']:.4f}
AUC-ROC:   {metrics['auc_roc']:.4f}
AUC-PR:    {metrics['auc_pr']:.4f}

CLASSIFICATION REPORT:
---------------------
{classification_report(y_test, y_pred, target_names=['Team 2 Wins', 'Team 1 Wins'])}

CONFUSION MATRIX:
----------------
{confusion_matrix(y_test, y_pred)}

PREDICTION CONFIDENCE ANALYSIS:
------------------------------
Average Confidence (Correct): {np.mean(y_pred_proba[(y_test == (y_pred_proba > 0.5).astype(int))]):.4f}
Average Confidence (Incorrect): {np.mean(y_pred_proba[~(y_test == (y_pred_proba > 0.5).astype(int))]):.4f}

"""
        
        # Add cross-validation results if training data is provided
        if X_train is not None and y_train is not None:
            cv_results = self.cross_validation_analysis(X_train, y_train)
            report += f"""
CROSS-VALIDATION RESULTS:
------------------------
"""
            for metric, results in cv_results.items():
                report += f"{metric.upper()}: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})\n"
        
        return report
    
    def save_evaluation_plots(self, X_test: np.ndarray, y_test: np.ndarray, 
                            X_train: np.ndarray = None, y_train: np.ndarray = None):
        """Generate and save all evaluation plots"""
        print("Generating evaluation plots...")
        
        # Get predictions
        _, y_pred, y_pred_proba = self.evaluate_performance(X_test, y_test)
        
        # Generate plots
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        self.plot_confusion_matrix_detailed(y_test, y_pred)
        self.plot_prediction_confidence(y_test, y_pred_proba)
        
        if X_train is not None and y_train is not None:
            self.plot_learning_curve(X_train, y_train)
            self.cross_validation_analysis(X_train, y_train)
        
        self.feature_importance_analysis()
        
        print("All evaluation plots saved!")


if __name__ == "__main__":
    from ml_model import ValorantMLModel
    from data_collector import ValorantDataCollector
    from data_preprocessor import ValorantDataPreprocessor
    from feature_engineering import ValorantFeatureEngineer
    
    # Load or train model
    print("Loading/Testing Valorant ML Model...")
    
    # This would typically load a pre-trained model
    # For demonstration, we'll show the evaluation framework structure
    print("Evaluation framework ready!")
    print("Use this class to evaluate your trained models.")
