# Visualization Guide for Student Score Prediction Project

This guide explains how to use the visualization features added to your machine learning project.

## Files Created

### 1. Visualization Modules
- `visualizations_fixed.py` - Main visualization module (saves plots as images)
- `visualizations.py` - Original module (tries to display plots directly)

### 2. Enhanced Main Scripts
- `student_score_prediction_visuals_fixed.py` - Main script with visualizations
- `student_score_prediction_with_visuals.py` - Original version with display

### 3. Output Files
- `visualization_outputs/` - Folder containing all generated images
- `visualization_report.html` - HTML report with all visualizations
- `view_visualizations.py` - Script to view visualizations

## How to Use

### Option 1: Run the Enhanced Main Script (Recommended)
```bash
cd student_score_prediction
python3 student_score_prediction_visuals_fixed.py
```

When prompted, enter student details:
```
Hours studied: 7.5
Attendance percentage: 90
Previous marks: 78
Assignments completed: 8
```

### Option 2: View Generated Visualizations
```bash
cd student_score_prediction
python3 view_visualizations.py
```

Then choose option 1 to open the HTML report in your browser.

### Option 3: Open HTML Report Directly
Open `student_score_prediction/visualization_report.html` in any web browser.

## Visualizations Generated

The system creates 9 different visualizations:

### 1. Data Analysis Visualizations
1. **Correlation Heatmap** - Shows relationships between all features
2. **Feature Distributions** - Histograms of each feature with statistics
3. **Feature vs Target** - Scatter plots showing how each feature affects final score

### 2. Model Performance Visualizations
4. **Model Comparison** - Bar charts comparing R² and MAE of all models
5. **Feature Importance (Random Forest)** - Which features matter most for tree models
6. **Residual Analysis (Linear Regression)** - Error analysis for the best model

### 3. Additional Visualizations
7. **Feature Importance (Decision Tree)**
8. **Residuals (Decision Tree)**
9. **Residuals (Random Forest)**

## Troubleshooting

### Problem: Graphs don't open/show
**Solution**: The fixed version saves graphs as image files instead of trying to display them. Check the `visualization_outputs/` folder for PNG files.

### Problem: Can't see images
**Solution**:
1. Run `python3 view_visualizations.py` and choose option 1
2. Or open `visualization_report.html` directly in a browser
3. Or check the `visualization_outputs/` folder manually

### Problem: Missing visualization_outputs folder
**Solution**: Run the main script first:
```bash
python3 student_score_prediction_visuals_fixed.py
```

## For Viva/Presentation

1. **Show HTML Report**: Open `visualization_report.html` - it has all graphs with explanations
2. **Explain Key Findings**:
   - Hours studied has strongest correlation (0.626)
   - Linear Regression performs best (R² = 0.9999)
   - Clear positive relationships in scatter plots
3. **Demonstrate Prediction**: Show how the model predicts scores for new students

## Technical Details

- **Libraries Used**: matplotlib, seaborn, pandas, numpy
- **Image Format**: PNG (high quality, 300 DPI)
- **File Location**: All images saved in `visualization_outputs/` folder
- **Report Format**: HTML with responsive design

## Customization

You can modify `visualizations_fixed.py` to:
- Change colors/styles
- Add new plot types
- Adjust figure sizes
- Change output format (PDF, SVG, etc.)

## Quick Start

Just run:
```bash
cd student_score_prediction
python3 student_score_prediction_visuals_fixed.py
# Enter student details when prompted
# Then open visualization_report.html in browser
```

Your project now has complete visualization capabilities for your viva presentation!
