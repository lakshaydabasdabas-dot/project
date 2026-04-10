"""
Visualization module for Student Exam Score Prediction System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_exam_scores.csv"


def load_data():
    """Load the student dataset"""
    return pd.read_csv(DATA_FILE)


def plot_correlation_heatmap(data):
    """Create a correlation heatmap for all features"""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    correlation_matrix = data.corr(numeric_only=True)
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap of Student Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(data):
    """Create scatter plots for each feature vs final exam score"""
    features = ['hours_studied', 'attendance_percentage', 
                'previous_marks', 'assignments_completed']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        ax.scatter(data[feature], data['final_exam_score'], 
                  alpha=0.6, edgecolors='w', s=50)
        
        # Add trend line
        z = np.polyfit(data[feature], data['final_exam_score'], 1)
        p = np.poly1d(z)
        ax.plot(data[feature], p(data[feature]), 
                color='red', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Final Exam Score', fontsize=12)
        ax.set_title(f'{feature.replace("_", " ").title()} vs Final Score', 
                     fontsize=14, pad=10)
        
        # Add correlation coefficient
        corr = data[feature].corr(data['final_exam_score'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Feature Relationships with Final Exam Score', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_distribution(data):
    """Create distribution plots for all features"""
    features = ['hours_studied', 'attendance_percentage', 
                'previous_marks', 'assignments_completed', 'final_exam_score']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Create histogram with KDE
        sns.histplot(data[feature], kde=True, ax=ax, bins=15)
        
        # Add vertical line for mean
        mean_val = data[feature].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.1f}')
        
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', 
                     fontsize=13, pad=10)
        ax.legend()
        
        # Add statistics
        stats_text = f"Min: {data[feature].min():.1f}\n" \
                    f"Max: {data[feature].max():.1f}\n" \
                    f"Std: {data[feature].std():.1f}"
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide the empty subplot
    axes[-1].axis('off')
    
    plt.suptitle('Distribution of Student Features', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(model_results):
    """Create bar chart comparing model performance"""
    model_names = [result['name'] for result in model_results]
    r2_scores = [result['r2'] for result in model_results]
    mae_scores = [result['mae'] for result in model_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² Score comparison
    bars1 = ax1.bar(model_names, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Comparison: R² Score (Higher is Better)', 
                  fontsize=14, pad=15)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11)
    
    # MAE comparison
    bars2 = ax2.bar(model_names, mae_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax2.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
    ax2.set_title('Model Comparison: MAE (Lower is Better)', 
                  fontsize=14, pad=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Machine Learning Model Performance Comparison', 
                 fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(tree_model, feature_names):
    """Plot feature importance for tree-based models"""
    if hasattr(tree_model, 'feature_importances_'):
        importances = tree_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(sorted_features)), sorted_importances, 
                      color=sns.color_palette("husl", len(sorted_features)))
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title('Feature Importance in Tree-Based Model', fontsize=16, pad=20)
        plt.xticks(range(len(sorted_features)), 
                  [f.replace('_', '\n').title() for f in sorted_features], 
                  rotation=0, fontsize=11)
        
        # Add value labels on bars
        for bar, importance in zip(bars, sorted_importances):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't have feature importances attribute.")


def plot_residuals(y_true, y_pred, model_name):
    """Plot residuals to check model errors"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', s=50)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title(f'Residuals vs Predicted - {model_name}', fontsize=14, pad=15)
    ax1.grid(alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Distribution of Residuals - {model_name}', fontsize=14, pad=15)
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'Residual Analysis for {model_name}', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


def run_all_visualizations():
    """Run all visualizations for the project"""
    print("Loading data...")
    data = load_data()
    
    print("\n1. Creating correlation heatmap...")
    plot_correlation_heatmap(data)
    
    print("\n2. Creating feature vs target plots...")
    plot_feature_vs_target(data)
    
    print("\n3. Creating distribution plots...")
    plot_distribution(data)
    
    print("\nVisualizations completed!")


if __name__ == "__main__":
    run_all_visualizations()