# Student Exam Score Prediction System

This project is a simple machine learning system that predicts a student's final exam score.

It keeps the same pipeline as a basic score prediction project:

- Student Data Analysis
- Data Preparation
- Performance Insights
- Score Prediction Model
- Model Evaluation
- Final Score Estimation

## Project Idea

The goal is to predict `final_exam_score` using these input features:

- `hours_studied`
- `attendance_percentage`
- `previous_marks`
- `assignments_completed`

This project is easy to explain in a viva because each feature has a clear academic meaning:

- If a student studies more, marks usually improve.
- If attendance is high, the student understands more classroom teaching.
- If previous marks are good, the student often has a strong academic base.
- If assignments are completed regularly, practice improves performance.

## Files

- `generate_dataset.py`: Creates a student dataset with 120 rows.
- `student_exam_scores.csv`: The dataset used for training and testing.
- `student_score_prediction.py`: Runs the full machine learning pipeline.

## How the Project Works

### 1. Student Data Analysis

We first inspect the dataset shape, sample rows, and summary values.

Simple viva explanation:
This step helps us understand what type of student data we have before training the model.

### 2. Data Preparation

We check for missing values and duplicate rows.

Simple viva explanation:
If the data is not clean, the model can learn wrong patterns.

### 3. Performance Insights

We study how each feature is related to the final exam score.

Simple viva explanation:
This helps us explain whether marks rise when study hours, attendance, or assignment completion increase.

### 4. Score Prediction Model

We train three models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Simple viva explanation:
We compare simple and advanced models to see which one predicts student marks better.

### 5. Model Evaluation

We use:

- `MAE` to check average prediction error
- `RMSE` to check overall prediction error
- `R2 Score` to see how well the model fits the data

Simple viva explanation:
Lower error means better prediction. Higher R2 means the model understands the pattern better.

### 6. Final Score Estimation

We give a new student's details and estimate the final exam score.

Simple viva explanation:
This is the practical output of the project.

## Short Conclusion for Viva

The best model is the one with the highest `R2 Score` and lower error values.

If `Linear Regression` performs best, a simple explanation is:

The dataset follows a direct pattern. When study hours, attendance, previous marks, and assignment completion increase, the final score also generally increases. So Linear Regression works well.

If `Random Forest` performs best, a simple explanation is:

It handles mixed score patterns better by combining many small decision trees.

## Run the Project

```bash
python3 generate_dataset.py
python3 student_score_prediction.py
```
