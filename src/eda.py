"""
Exploratory Data Analysis (EDA) Module
=======================================
Creates comprehensive visualizations for the Heart Disease dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_data


def set_style():
    """Set consistent plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def plot_target_distribution(df, output_dir):
    """Plot the distribution of target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    colors = ['#2ecc71', '#e74c3c']
    counts = df['target'].value_counts()
    axes[0].bar(['No Disease (0)', 'Heart Disease (1)'], counts.values, color=colors, 
                edgecolor='white', linewidth=2)
    axes[0].set_title('Heart Disease Distribution', fontweight='bold', fontsize=16)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold', fontsize=14)
    
    # Pie chart
    axes[1].pie(counts.values, labels=['No Disease', 'Heart Disease'], colors=colors,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14},
                explode=(0.05, 0.05), shadow=True)
    axes[1].set_title('Target Class Proportion', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Target distribution plot saved")


def plot_age_distribution(df, output_dir):
    """Plot age distribution by heart disease status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(df[df['target']==0]['age'], bins=20, alpha=0.7, label='No Disease', 
                 color='#2ecc71', edgecolor='white')
    axes[0].hist(df[df['target']==1]['age'], bins=20, alpha=0.7, label='Heart Disease', 
                 color='#e74c3c', edgecolor='white')
    axes[0].set_title('Age Distribution by Heart Disease', fontweight='bold', fontsize=16)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Box plot
    sns.boxplot(x='target', y='age', data=df, ax=axes[1], 
                palette=['#2ecc71', '#e74c3c'])
    axes[1].set_xticklabels(['No Disease', 'Heart Disease'])
    axes[1].set_title('Age Box Plot by Disease Status', fontweight='bold', fontsize=16)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Age')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Age distribution plot saved")


def plot_correlation_heatmap(df, output_dir):
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu_r', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Correlation heatmap saved")


def plot_feature_distributions(df, output_dir):
    """Plot distributions of all continuous features."""
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(continuous_features):
        sns.histplot(data=df, x=feature, hue='target', kde=True, ax=axes[i],
                     palette=['#2ecc71', '#e74c3c'], alpha=0.6)
        axes[i].set_title(f'{feature.upper()} Distribution', fontweight='bold')
        axes[i].legend(['No Disease', 'Heart Disease'])
    
    # Remove extra subplot
    axes[-1].set_visible(False)
    
    plt.suptitle('Continuous Feature Distributions by Heart Disease Status', 
                 fontweight='bold', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature distributions plot saved")


def plot_categorical_features(df, output_dir):
    """Plot categorical feature distributions."""
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    feature_labels = {
        'sex': {0: 'Female', 1: 'Male'},
        'cp': {0: 'Typical', 1: 'Atypical', 2: 'Non-anginal', 3: 'Asymptomatic'},
        'fbs': {0: '≤120', 1: '>120'},
        'restecg': {0: 'Normal', 1: 'ST-T Abnorm.', 2: 'LV Hyper.'},
        'exang': {0: 'No', 1: 'Yes'},
        'slope': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'},
        'ca': {0: '0', 1: '1', 2: '2', 3: '3'},
        'thal': {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible', 3: 'Defect'}
    }
    
    for i, feature in enumerate(categorical_features):
        ct = pd.crosstab(df[feature], df['target'])
        ct.plot(kind='bar', ax=axes[i], color=['#2ecc71', '#e74c3c'], 
                edgecolor='white', linewidth=1)
        axes[i].set_title(f'{feature.upper()}', fontweight='bold')
        axes[i].legend(['No Disease', 'Heart Disease'], fontsize=8)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=0)
    
    plt.suptitle('Categorical Feature Analysis by Heart Disease', 
                 fontweight='bold', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Categorical features plot saved")


def plot_gender_analysis(df, output_dir):
    """Detailed gender-based analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gender distribution
    gender_counts = df['sex'].value_counts()
    axes[0].bar(['Female', 'Male'], gender_counts.values, color=['#e91e63', '#2196f3'],
                edgecolor='white', linewidth=2)
    axes[0].set_title('Gender Distribution', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Count')
    
    # Heart disease by gender
    ct = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='white')
    axes[1].set_xticklabels(['Female', 'Male'], rotation=0)
    axes[1].set_title('Heart Disease Rate by Gender (%)', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('Percentage')
    axes[1].legend(['No Disease', 'Heart Disease'])
    
    # Age distribution by gender and disease
    for sex, color, label in [(0, '#e91e63', 'Female'), (1, '#2196f3', 'Male')]:
        subset = df[(df['sex'] == sex) & (df['target'] == 1)]
        axes[2].hist(subset['age'], bins=15, alpha=0.6, label=f'{label} with Disease', color=color)
    axes[2].set_title('Age of Heart Disease Patients by Gender', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Gender analysis plot saved")


def plot_pairplot(df, output_dir):
    """Create pair plot for key features."""
    key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
    
    g = sns.pairplot(df[key_features], hue='target', 
                     palette=['#2ecc71', '#e74c3c'],
                     diag_kind='hist', plot_kws={'alpha': 0.6},
                     height=2.5)
    g.fig.suptitle('Pair Plot of Key Features', fontweight='bold', fontsize=18, y=1.02)
    
    plt.savefig(os.path.join(output_dir, 'pairplot.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("✅ Pair plot saved")


def run_eda(output_dir='outputs'):
    """Run complete EDA pipeline."""
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    set_style()
    
    # Load data
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    df = load_data()
    print(f"\n📊 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Generate all plots
    print("\n🎨 Generating visualizations...\n")
    
    plot_target_distribution(df, output_dir)
    plot_age_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_feature_distributions(df, output_dir)
    plot_categorical_features(df, output_dir)
    plot_gender_analysis(df, output_dir)
    plot_pairplot(df, output_dir)
    
    print(f"\n🎉 All EDA visualizations saved to '{output_dir}/' directory!")
    print("=" * 60)


if __name__ == "__main__":
    # Run from project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    run_eda()
