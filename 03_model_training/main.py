"""
Step 3: Model Training Module
============================

This module trains models for schedule generation:
- Teacher assignment model
- Schedule quality predictor
- Constraint satisfaction predictor

Usage:
    python 03_model_training/main.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import joblib
import json
from typing import Dict, Any

# Suppress sklearn warnings effectively
warnings.filterwarnings("ignore")
warnings.filterwarnings("once", module="sklearn.utils.validation")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelTrainer:
    def __init__(self, config_path: str):
        """Initialize model trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create output directory
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        
        # Load features
        self.compatibility_features = pd.read_csv(
            os.path.join(self.config['paths']['features_dir'], 'teacher_class_compatibility.csv')
        )
        self.constraint_features = pd.read_csv(
            os.path.join(self.config['paths']['features_dir'], 'constraint_satisfaction.csv')
        )
        self.quality_features = pd.read_csv(
            os.path.join(self.config['paths']['features_dir'], 'quality_prediction.csv')
        )
        
        # Initialize models
        self.models = {}
        self.model_metrics = {}

    def train_teacher_assignment_model(self):
        """Train model for teacher assignment prediction."""
        print("Training teacher assignment model...")
        
        # Prepare features and target
        feature_cols = [
            'grade', 'grade_priority', 'gender_compatible', 'building_match', 'primary_subject_match',
            'secondary_subject_capability', 'subject_capability', 'subject_combination_score',
            'teacher_availability', 'daily_class_required'
        ]
        
        X = self.compatibility_features[feature_cols].values
        y = self.compatibility_features['subject_capability'].values  # Use subject capability as target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['ml']['teacher_assignment']['test_size'],
            random_state=self.config['ml']['teacher_assignment']['random_state']
        )
        
        # Train model
        model = LogisticRegression(
            max_iter=self.config['ml']['teacher_assignment']['max_iter'],
            C=self.config['ml']['teacher_assignment']['C'],
            random_state=self.config['ml']['teacher_assignment']['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store model and metrics
        self.models['teacher_assignment'] = model
        self.model_metrics['teacher_assignment'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': dict(zip(feature_cols, model.coef_[0]))
        }
        
        print(f"  Teacher assignment model trained")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")

    def train_schedule_quality_model(self):
        """Train model for schedule quality prediction."""
        print("Training schedule quality model...")
        
        # Prepare features and target
        feature_cols = [
            'grade', 'total_periods', 'unique_subjects', 'available_teachers',
            'available_rooms', 'teacher_utilization', 'room_utilization'
        ]
        
        X = self.quality_features[feature_cols].values
        y = self.quality_features['quality_score'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['ml']['schedule_quality']['test_size'],
            random_state=self.config['ml']['schedule_quality']['random_state']
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=self.config['ml']['schedule_quality']['n_estimators'],
            max_depth=self.config['ml']['schedule_quality']['max_depth'],
            random_state=self.config['ml']['schedule_quality']['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store model and metrics
        self.models['schedule_quality'] = model
        self.model_metrics['schedule_quality'] = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
        
        print(f"  Schedule quality model trained")
        print(f"    R² Score: {r2:.4f}")
        print(f"    MSE: {mse:.4f}")
        print(f"    RMSE: {rmse:.4f}")

    def train_constraint_satisfaction_model(self):
        """Train model for constraint satisfaction prediction."""
        print("Training constraint satisfaction model...")
        
        # Prepare features and target
        feature_cols = [
            'gender_compatible', 'teacher_available', 'room_type_compatible',
            'building_match', 'primary_subject_match'
        ]
        
        X = self.constraint_features[feature_cols].values
        y = self.constraint_features['feasible'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['ml']['conflict_resolution']['test_size'],
            random_state=self.config['ml']['conflict_resolution']['random_state']
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=self.config['ml']['conflict_resolution']['n_estimators'],
            max_depth=self.config['ml']['conflict_resolution']['max_depth'],
            random_state=self.config['ml']['conflict_resolution']['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store model and metrics
        self.models['constraint_satisfaction'] = model
        self.model_metrics['constraint_satisfaction'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
        
        print(f"  Constraint satisfaction model trained")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")

    def save_models(self):
        """Save all trained models and their metrics."""
        print("Saving models and metrics...")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(self.config['paths']['models_dir'], f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.config['paths']['models_dir'], 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        print(f"  Models saved to: {self.config['paths']['models_dir']}")
        print(f"  Metrics saved to: {metrics_path}")

    def print_model_summary(self):
        """Print summary of all trained models."""
        print("\nModel Training Summary")
        print("=" * 50)
        
        for model_name, metrics in self.model_metrics.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            for metric_name, value in metrics.items():
                if metric_name != 'feature_importance':
                    print(f"  {metric_name}: {value:.4f}")
            
            # Print top 3 most important features
            if 'feature_importance' in metrics:
                importance = metrics['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                print("  Top features:")
                for feature, importance in sorted_features[:3]:
                    print(f"    {feature}: {importance:.4f}")

def main():
    """Main function for model training."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')
    
    print("Starting Step 3: Model Training")
    print("=" * 50)
    
    # Initialize model trainer
    trainer = ModelTrainer(config_path)
    
    # Train all models
    trainer.train_teacher_assignment_model()
    trainer.train_schedule_quality_model()
    trainer.train_constraint_satisfaction_model()
    
    # Save models and metrics
    trainer.save_models()
    
    # Print summary
    trainer.print_model_summary()
    
    print("\nStep 3 completed successfully!")
    print(" Ready for Step 4: Schedule Generation")

if __name__ == "__main__":
    main()
