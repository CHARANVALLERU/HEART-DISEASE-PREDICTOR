"""
Model Training & Evaluation Module
====================================
Trains multiple ML models, evaluates performance, and saves the best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_data, explore_data, preprocess_data


def get_models():
    """
    Define and return all ML models to train.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42, C=1.0
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=8
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1
        )
    }
    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Train all models and collect evaluation metrics.
    
    Returns:
        results_df: DataFrame with all metrics
        trained_models: dict of trained model objects
    """
    results = []
    trained_models = {}
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        print(f"   ✅ Accuracy:  {accuracy:.4f}")
        print(f"   📊 Precision: {precision:.4f}")
        print(f"   📊 Recall:    {recall:.4f}")
        print(f"   📊 F1-Score:  {f1:.4f}")
        print(f"   📊 ROC-AUC:   {roc_auc:.4f}")
        print(f"   📊 CV Score:  {cv_mean:.4f} (±{cv_std:.4f})")
        
        print(f"\n   Classification Report for {name}:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['No Disease', 'Heart Disease']))
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    return results_df, trained_models


def plot_model_comparison(results_df, output_dir='outputs'):
    """Plot model accuracy comparison bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
    bars = axes[0].barh(results_df['Model'], results_df['Accuracy'] * 100, 
                        color=colors, edgecolor='white', linewidth=2)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=16)
    axes[0].set_xlim(70, 100)
    
    for bar, acc in zip(bars, results_df['Accuracy']):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                     f'{acc*100:.1f}%', va='center', fontweight='bold')
    
    # Multi-metric comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df['Model']))
    width = 0.18
    
    for i, metric in enumerate(metrics):
        axes[1].bar(x + i * width, results_df[metric] * 100, width, 
                    label=metric, alpha=0.85)
    
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Score (%)')
    axes[1].set_title('Multi-Metric Model Comparison', fontweight='bold', fontsize=16)
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(results_df['Model'], rotation=30, ha='right')
    axes[1].legend()
    axes[1].set_ylim(70, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Model comparison plot saved")


def plot_confusion_matrices(trained_models, X_test, y_test, output_dir='outputs'):
    """Plot confusion matrices for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No Disease', 'Heart Disease'],
                    yticklabels=['No Disease', 'Heart Disease'],
                    cbar=False, linewidths=2, linecolor='white')
        axes[i].set_title(f'{name}', fontweight='bold', fontsize=14)
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
    
    plt.suptitle('Confusion Matrices - All Models', fontweight='bold', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrices plot saved")


def plot_roc_curves(trained_models, X_test, y_test, output_dir='outputs'):
    """Plot ROC curves for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
    
    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                    color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=18)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ ROC curves plot saved")


def plot_feature_importance(trained_models, feature_names, output_dir='outputs'):
    """Plot feature importance from Random Forest and Gradient Boosting."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, model_name in enumerate(['Random Forest', 'Gradient Boosting']):
        if model_name in trained_models:
            model = trained_models[model_name]
            importances = model.feature_importances_
            indices = np.argsort(importances)
            
            colors = plt.cm.RdYlGn(importances[indices] / importances.max())
            
            axes[idx].barh(range(len(indices)), importances[indices], color=colors,
                          edgecolor='white', linewidth=1)
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels([feature_names[i] for i in indices])
            axes[idx].set_title(f'{model_name}\nFeature Importance', fontweight='bold', fontsize=14)
            axes[idx].set_xlabel('Importance Score')
    
    plt.suptitle('Feature Importance Analysis', fontweight='bold', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance plot saved")


def plot_cv_scores(results_df, output_dir='outputs'):
    """Plot cross-validation scores."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
    
    ax.bar(results_df['Model'], results_df['CV Mean'] * 100, 
           yerr=results_df['CV Std'] * 100, color=colors, 
           edgecolor='white', linewidth=2, capsize=5, error_kw={'linewidth': 2})
    
    ax.set_ylabel('Cross-Validation Accuracy (%)')
    ax.set_title('5-Fold Cross-Validation Scores', fontweight='bold', fontsize=16)
    ax.set_ylim(70, 100)
    plt.xticks(rotation=30, ha='right')
    
    for i, (cv_mean, cv_std) in enumerate(zip(results_df['CV Mean'], results_df['CV Std'])):
        ax.text(i, cv_mean * 100 + cv_std * 100 + 1, f'{cv_mean*100:.1f}%', 
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_scores.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Cross-validation scores plot saved")


def save_best_model(results_df, trained_models, models_dir='models'):
    """Save the best performing model."""
    os.makedirs(models_dir, exist_ok=True)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_accuracy = results_df.iloc[0]['Accuracy']
    
    model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    
    # Save model info
    info_path = os.path.join(models_dir, 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Precision: {results_df.iloc[0]['Precision']:.4f}\n")
        f.write(f"Recall: {results_df.iloc[0]['Recall']:.4f}\n")
        f.write(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}\n")
        f.write(f"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}\n")
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Model saved to: {model_path}")
    
    return best_model_name, best_model


def run_training_pipeline():
    """Execute the complete training pipeline."""
    print("\n" + "=" * 70)
    print("❤️  HEART DISEASE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load & Explore data
    print("\n📥 Step 1: Loading Data...")
    df = load_data()
    df = explore_data(df)
    
    # Step 2: Preprocess data
    print("\n⚙️ Step 2: Preprocessing Data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Step 3: Get models
    print("\n🧠 Step 3: Initializing Models...")
    models = get_models()
    print(f"   Models to train: {list(models.keys())}")
    
    # Step 4: Train & Evaluate
    print("\n🚀 Step 4: Training & Evaluating...")
    results_df, trained_models = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    
    # Step 5: Print Results Summary
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Step 6: Generate Plots
    print("\n🎨 Step 5: Generating Plots...")
    plot_model_comparison(results_df)
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_feature_importance(trained_models, feature_names)
    plot_cv_scores(results_df)
    
    # Step 7: Save Best Model
    print("\n💾 Step 6: Saving Best Model...")
    best_name, best_model = save_best_model(results_df, trained_models)
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Outputs saved to 'outputs/' directory")
    print(f"📁 Model saved to 'models/' directory")
    print(f"🏆 Best model: {best_name}")
    print(f"\n🚀 Run 'streamlit run src/app.py' to launch the web app!")
    
    return results_df, trained_models


if __name__ == "__main__":
    # Run from project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    run_training_pipeline()
