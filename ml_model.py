"""
Machine Learning model for Valorant match outcome prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from config import MODEL_CONFIG


class ValorantMLModel:
    """Machine Learning model for predicting Valorant match outcomes"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training and testing"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, any]:
        """Train multiple models and select the best one"""
        print("Training multiple models...")
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG['random_state'],
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG['random_state'],
                learning_rate=0.1,
                max_depth=6
            ),
            'Logistic Regression': LogisticRegression(
                random_state=MODEL_CONFIG['random_state'],
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=MODEL_CONFIG['random_state'],
                probability=True,
                kernel='rbf'
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        return self.models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=MODEL_CONFIG['cv_folds'])
            
            results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print()
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict]) -> str:
        """Select the best model based on evaluation metrics"""
        best_score = -1
        best_model_name = None
        
        for name, metrics in results.items():
            # Use AUC as primary metric, fallback to accuracy
            score = metrics['auc'] if metrics['auc'] is not None else metrics['accuracy']
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name} (Score: {best_score:.4f})")
        return best_model_name
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> any:
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {self.best_model_name}...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10]
            }
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
        else:
            print("Hyperparameter tuning not implemented for this model type")
            return self.best_model
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update best model with tuned parameters
        self.best_model = grid_search.best_estimator_
        return self.best_model
    
    def plot_feature_importance(self, feature_names: List[str], top_n: int = 15):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.best_model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return
        
        # Get feature importance
        importance = self.best_model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{self.best_model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Team 2 Wins', 'Team 1 Wins'],
                   yticklabels=['Team 2 Wins', 'Team 1 Wins'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.best_model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename: str = None):
        """Save the trained model and scaler"""
        if filename is None:
            filename = f"valorant_model_{self.best_model_name.lower().replace(' ', '_')}"
        
        # Save model
        joblib.dump(self.best_model, f"{filename}.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, f"{filename}_scaler.joblib")
        
        # Save feature names
        with open(f"{filename}_features.txt", 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        print(f"Model saved as {filename}.joblib")
        print(f"Scaler saved as {filename}_scaler.joblib")
        print(f"Features saved as {filename}_features.txt")
    
    def load_model(self, filename: str):
        """Load a trained model and scaler"""
        self.best_model = joblib.load(f"{filename}.joblib")
        self.scaler = joblib.load(f"{filename}_scaler.joblib")
        
        # Load feature names
        with open(f"{filename}_features.txt", 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Model loaded from {filename}.joblib")
    
    def predict_match(self, team1_name: str, team2_name: str, 
                     team1_country: str = None, team2_country: str = None) -> Dict:
        """Predict the outcome of a specific match"""
        if self.best_model is None:
            raise ValueError("Model not trained or loaded")
        
        # Create features for prediction
        from feature_engineering import ValorantFeatureEngineer
        
        # This would need the team stats, region strength, and h2h stats
        # For now, we'll create a simple prediction interface
        print(f"Predicting match: {team1_name} vs {team2_name}")
        print("Note: This requires the feature engineering module with team statistics")
        
        # Placeholder prediction
        prediction = {
            'team1_name': team1_name,
            'team2_name': team2_name,
            'team1_win_probability': 0.5,
            'team2_win_probability': 0.5,
            'predicted_winner': 'Tie',
            'confidence': 0.0
        }
        
        return prediction
    
    def train_full_pipeline(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict:
        """Complete training pipeline"""
        print("Starting full training pipeline...")
        
        self.feature_names = feature_names
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Select best model
        best_model_name = self.select_best_model(results)
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train, y_train)
        
        # Final evaluation
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        final_accuracy = accuracy_score(y_test, y_pred)
        final_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nFinal Model Performance:")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"AUC: {final_auc:.4f}")
        
        # Plot results
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance(feature_names)
        
        # Save model
        self.save_model()
        
        return {
            'best_model': self.best_model_name,
            'accuracy': final_accuracy,
            'auc': final_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_labels': y_test
        }


if __name__ == "__main__":
    from data_collector import ValorantDataCollector
    from data_preprocessor import ValorantDataPreprocessor
    from feature_engineering import ValorantFeatureEngineer
    
    # Complete pipeline
    print("Starting Valorant ML Pipeline...")
    
    # Collect data
    collector = ValorantDataCollector()
    raw_data = collector.collect_all_data()
    
    # Preprocess data
    preprocessor = ValorantDataPreprocessor()
    df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)
    
    # Engineer features
    feature_engineer = ValorantFeatureEngineer(team_stats, region_strength, h2h_stats)
    X, y, metadata, feature_names = feature_engineer.prepare_ml_features(df)
    
    # Train model
    ml_model = ValorantMLModel()
    results = ml_model.train_full_pipeline(X, y, feature_names)
    
    print("Training completed!")
