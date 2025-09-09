"""
Baseline Models for Emotion Detection

This module implements baseline machine learning models for emotion classification:
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest (bonus)

These models are trained on extracted features (LBP or CNN) and serve as baselines
for comparison with deep learning approaches.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import joblib

logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    """
    Trainer for baseline machine learning models.
    
    Features:
    - Train SVM, Logistic Regression, and Random Forest models
    - Hyperparameter tuning with GridSearchCV
    - Cross-validation evaluation
    - Model persistence and loading
    - Performance comparison
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the baseline model trainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_params = {}
        
        # Define model configurations
        self.model_configs = {
            'svm': {
                'model': SVC(random_state=random_state),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l1', 'l2']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        logger.info("Initialized BaselineModelTrainer")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            X: Feature matrix
            y: Label array
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name}...")
        
        # Get model configuration
        config = self.model_configs[model_name]
        model = config['model']
        params = config['params']
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, params, cv=cv_folds, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model and parameters
        self.models[model_name] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(
            grid_search.best_estimator_, X_train, y_train, cv=cv_folds
        )
        
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv_folds: int = 5) -> pd.DataFrame:
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with training results for all models
        """
        all_results = []
        
        for model_name in self.model_configs.keys():
            try:
                results = self.train_single_model(model_name, X_train, y_train, cv_folds)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sort by best score
        results_df = results_df.sort_values('best_score', ascending=False)
        
        logger.info("Training completed for all models")
        return results_df
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results for each model
        """
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = model.score(X_test, y_test)
                
                # Classification report
                class_report = classification_report(
                    y_test, y_pred, 
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                results = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'prediction_probabilities': y_pred_proba,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix
                }
                
                evaluation_results[model_name] = results
                
                logger.info(f"{model_name} - Test Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return evaluation_results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on training scores.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.models:
            raise ValueError("No models have been trained yet")
        
        # Find model with highest training score
        best_model_name = max(self.best_params.keys(), 
                            key=lambda x: self.best_params[x].get('score', 0))
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, output_dir: str):
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = output_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to: {model_path}")
        
        # Save label encoder
        encoder_path = output_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save best parameters
        params_path = output_dir / "best_parameters.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(self.best_params, f)
        
        logger.info(f"All models saved to: {output_dir}")
    
    def load_models(self, model_dir: str):
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        # Load models
        for model_name in self.model_configs.keys():
            model_path = model_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model from: {model_path}")
        
        # Load label encoder
        encoder_path = model_dir / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        
        # Load best parameters
        params_path = model_dir / "best_parameters.pkl"
        if params_path.exists():
            with open(params_path, 'rb') as f:
                self.best_params = pickle.load(f)
        
        logger.info(f"Models loaded from: {model_dir}")
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Predicted labels
        """
        if model_name is None:
            model_name, model = self.get_best_model()
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        predictions = model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Get prediction probabilities using a trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Prediction probabilities
        """
        if model_name is None:
            model_name, model = self.get_best_model()
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"Model {model_name} does not support probability predictions")

def main():
    """Example usage of the baseline model trainer."""
    # This would be called with actual feature data
    # trainer = BaselineModelTrainer()
    # 
    # # Load features (example)
    # X, y, _ = load_features("data/processed/lbp_features.pkl")
    # 
    # # Prepare data
    # X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    # 
    # # Train all models
    # results_df = trainer.train_all_models(X_train, y_train)
    # print(results_df)
    # 
    # # Evaluate models
    # eval_results = trainer.evaluate_models(X_test, y_test)
    # 
    # # Save models
    # trainer.save_models("models/baseline")
    pass

if __name__ == "__main__":
    main()
