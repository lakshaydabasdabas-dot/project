#!/usr/bin/env python3
"""
Student Score Prediction System - All-in-One
Combines: data generation, prediction, visualization, and HTML report
"""

import csv
import random
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_exam_scores.csv"
OUTPUT_DIR = BASE_DIR / "visualization_outputs"
HTML_FILE = BASE_DIR / "visualization_report.html"

OUTPUT_FILE = "student_exam_scores.csv"
ROW_COUNT = 120
SEED = 42

# ============================================================================
# DATASET GENERATION
# ============================================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def build_score(hours_studied, attendance_percentage, previous_marks, assignments_completed):
    noise = random.randint(-6, 6)
    score = (
        8
        + (3.1 * hours_studied)
        + (0.18 * attendance_percentage)
        + (0.42 * previous_marks)
        + (1.7 * assignments_completed)
        + noise
    )
    return round(clamp(score, 0, 100), 2)


def generate_dataset():
    """Generate random student exam score dataset"""
    print("=" * 70)
    print("GENERATING DATASET")
    print("=" * 70)
    
    random.seed(SEED)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "hours_studied",
            "attendance_percentage",
            "previous_marks",
            "assignments_completed",
            "final_exam_score",
        ])

        for _ in range(ROW_COUNT):
            hours_studied = round(random.uniform(1.0, 10.0), 1)
            attendance_percentage = random.randint(20, 100)
            previous_marks = random.randint(0, 100)
            assignments_completed = random.randint(0, 10)
            final_exam_score = build_score(
                hours_studied,
                attendance_percentage,
                previous_marks,
                assignments_completed,
            )

            writer.writerow([
                hours_studied,
                attendance_percentage,
                previous_marks,
                assignments_completed,
                final_exam_score,
            ])
    
    print(f"✓ Generated {ROW_COUNT} student records in {OUTPUT_FILE}")

# ============================================================================
# DATA ANALYSIS
# ============================================================================

def load_data():
    return pd.read_csv(DATA_FILE)


def analyze_data(data):
    """Print data analysis"""
    print("\n" + "=" * 70)
    print("DATA ANALYSIS")
    print("=" * 70)
    print(f"Dataset shape: {data.shape}")
    print(f"\nFirst 5 rows:\n{data.head()}")
    print(f"\nColumn summary:\n{data.describe()}")
    print(f"\nMissing values:\n{data.isnull().sum()}")
    print(f"\nCorrelation with final_exam_score:\n{data.corr(numeric_only=True)['final_exam_score'].sort_values(ascending=False)}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(fig, filename):
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")
    return filepath


def plot_correlation_heatmap(data):
    fig = plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Student Features', fontsize=16, pad=20)
    plt.tight_layout()
    return save_plot(fig, "correlation_heatmap.png")


def plot_feature_vs_target(data):
    features = ['hours_studied', 'attendance_percentage', 
                'previous_marks', 'assignments_completed']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        ax.scatter(data[feature], data['final_exam_score'], alpha=0.6, edgecolors='w', s=50)
        z = np.polyfit(data[feature], data['final_exam_score'], 1)
        p = np.poly1d(z)
        ax.plot(data[feature], p(data[feature]), color='red', linewidth=2, alpha=0.8)
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Final Exam Score', fontsize=12)
        ax.set_title(f'{feature.replace("_", " ").title()} vs Final Score', fontsize=14, pad=10)
        corr = data[feature].corr(data['final_exam_score'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Feature Relationships with Final Exam Score', fontsize=16, y=1.02)
    plt.tight_layout()
    return save_plot(fig, "feature_vs_target.png")


def plot_distribution(data):
    features = ['hours_studied', 'attendance_percentage', 
                'previous_marks', 'assignments_completed', 'final_exam_score']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.histplot(data[feature], kde=True, ax=ax, bins=15)
        mean_val = data[feature].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=13, pad=10)
        ax.legend()
        stats_text = f"Min: {data[feature].min():.1f}\nMax: {data[feature].max():.1f}\nStd: {data[feature].std():.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].axis('off')
    plt.suptitle('Distribution of Student Features', fontsize=16, y=1.02)
    plt.tight_layout()
    return save_plot(fig, "feature_distributions.png")


def plot_model_comparison(model_results):
    model_names = [result['name'] for result in model_results]
    r2_scores = [result['r2'] for result in model_results]
    mae_scores = [result['mae'] for result in model_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bars1 = ax1.bar(model_names, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Comparison: R² Score (Higher is Better)', fontsize=14, pad=15)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=11)
    
    bars2 = ax2.bar(model_names, mae_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax2.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
    ax2.set_title('Model Comparison: MAE (Lower is Better)', fontsize=14, pad=15)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Machine Learning Model Performance Comparison', fontsize=16, y=1.05)
    plt.tight_layout()
    return save_plot(fig, "model_comparison.png")


def plot_feature_importance(tree_model, feature_names):
    if hasattr(tree_model, 'feature_importances_'):
        importances = tree_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        fig = plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(sorted_features)), sorted_importances,
                      color=sns.color_palette("husl", len(sorted_features)))
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title('Feature Importance in Tree-Based Model', fontsize=16, pad=20)
        plt.xticks(range(len(sorted_features)),
                  [f.replace('_', '\n').title() for f in sorted_features], rotation=0, fontsize=11)
        for bar, importance in zip(bars, sorted_importances):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        model_name = tree_model.__class__.__name__
        return save_plot(fig, f"feature_importance_{model_name}.png")
    return None


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', s=50)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title(f'Residuals vs Predicted - {model_name}', fontsize=14, pad=15)
    ax1.grid(alpha=0.3)
    
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Distribution of Residuals - {model_name}', fontsize=14, pad=15)
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'Residual Analysis for {model_name}', fontsize=16, y=1.05)
    plt.tight_layout()
    return save_plot(fig, f"residuals_{model_name.replace(' ', '_')}.png")


# ============================================================================
# MODEL TRAINING & PREDICTION
# ============================================================================

def train_models(data):
    """Train ML models and return results"""
    feature_columns = ["hours_studied", "attendance_percentage", "previous_marks", "assignments_completed"]
    x = data[feature_columns]
    y = data["final_exam_score"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42, max_depth=5)),
        ("Random Forest", RandomForestRegressor(random_state=42, n_estimators=200, max_depth=6)),
    ]
    
    results = []
    for model_name, model in models:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        
        results.append({
            "name": model_name,
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "predictions": predictions,
            "y_test": y_test
        })
        
        print(f"  {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_correlation_heatmap(data)
    plot_feature_vs_target(data)
    plot_distribution(data)
    plot_model_comparison(results)
    
    for result in results:
        if result["name"] in ["Decision Tree", "Random Forest"]:
            plot_feature_importance(result["model"], feature_columns)
        plot_residuals(result["y_test"], result["predictions"], result["name"])
    
    best_model = max(results, key=lambda r: r["r2"])
    print(f"\n✓ Best model: {best_model['name']} (R² = {best_model['r2']:.4f})")
    
    return best_model, results


def predict_student_score(best_model):
    """Get user input and predict score"""
    print("\n" + "=" * 70)
    print("PREDICT STUDENT SCORE")
    print("=" * 70)
    print("Enter student details:")
    
    hours_studied = float(input("Hours studied (1-10): "))
    attendance = float(input("Attendance % (20-100): "))
    previous_marks = float(input("Previous marks (0-100): "))
    assignments = float(input("Assignments completed (0-10): "))
    
    student_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "attendance_percentage": attendance,
        "previous_marks": previous_marks,
        "assignments_completed": assignments
    }])
    
    predicted_score = best_model["model"].predict(student_data)[0]
    predicted_score = clamp(predicted_score, 0, 100)
    
    print(f"\n>>> Predicted Final Exam Score: {predicted_score:.2f}")
    
    return predicted_score

# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    import base64
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_report(results):
    """Generate HTML report with all visualizations"""
    print("\nGenerating HTML report...")
    
    images = [
        ("correlation_heatmap.png", "Correlation Heatmap"),
        ("feature_vs_target.png", "Features vs Final Score"),
        ("feature_distributions.png", "Feature Distributions"),
        ("model_comparison.png", "Model Comparison"),
        ("feature_importance_RandomForestRegressor.png", "Feature Importance (Random Forest)"),
        ("feature_importance_DecisionTreeRegressor.png", "Feature Importance (Decision Tree)"),
        ("residuals_Linear_Regression.png", "Residuals - Linear Regression"),
        ("residuals_Decision_Tree.png", "Residuals - Decision Tree"),
        ("residuals_Random_Forest.png", "Residuals - Random Forest"),
    ]
    
    html_sections = []
    for img_file, title in images:
        img_path = OUTPUT_DIR / img_file
        if img_path.exists():
            b64 = get_image_base64(img_path)
            ext = img_file.split('.')[-1]
            html_sections.append(f'''
            <div class="chart-container">
                <h2>{title}</h2>
                <img src="data:image/{ext};base64,{b64}" alt="{title}">
            </div>
            ''')
    
    # Model results table
    results_table = ""
    for r in results:
        results_table += f'''
        <tr>
            <td>{r['name']}</td>
            <td>{r['mae']:.2f}</td>
            <td>{r['rmse']:.2f}</td>
            <td>{r['r2']:.4f}</td>
        </tr>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Score Prediction - Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #333; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 40px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
        h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .subtitle {{ font-size: 1.2rem; opacity: 0.9; }}
        .section {{ background: white; border-radius: 10px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #667eea; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart-container {{ margin: 30px 0; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-container h2 {{ color: #667eea; margin-bottom: 15px; }}
        .chart-container img {{ width: 100%; max-width: 900px; height: auto; border-radius: 5px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-card h3 {{ font-size: 2rem; margin-bottom: 5px; }}
        .stat-card p {{ opacity: 0.9; }}
        footer {{ text-align: center; padding: 20px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Student Score Prediction</h1>
            <p class="subtitle">Analysis & Visualization Report</p>
        </header>
        
        <div class="section">
            <h2>Model Performance</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R² Score</th>
                </tr>
                {results_table}
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            {''.join(html_sections)}
        </div>
        
        <footer>
            <p>Generated by Student Score Prediction System</p>
        </footer>
    </div>
</body>
</html>'''
    
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ Saved: {HTML_FILE.name}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STUDENT SCORE PREDICTION SYSTEM - ALL IN ONE")
    print("=" * 70)
    
    # Step 1: Generate dataset
    generate_dataset()
    
    # Step 2: Load and analyze data
    data = load_data()
    analyze_data(data)
    
    # Step 3: Train models and create visualizations
    best_model, results = train_models(data)
    
    # Step 4: Generate HTML report
    generate_html_report(results)
    
    # Step 5: Predict student score
    predict_student_score(best_model)
    
    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)
    print(f"Open {HTML_FILE.name} in browser to view the report with all graphs.")


if __name__ == "__main__":
    main()
