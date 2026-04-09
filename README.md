# Student Exam Score Prediction System

This repository contains a simple machine learning mini-project that predicts a student's final exam score using academic performance indicators.

The project is designed to be easy to demonstrate in a viva because:

- the problem statement is simple and practical
- the input features are easy to explain
- the code follows a clear step-by-step ML pipeline
- the final output is an actual score prediction for a new student

The main working project is inside `student_score_prediction/`.

## Project Objective

The objective of this project is to predict `final_exam_score` from the following input features:

- `hours_studied`
- `attendance_percentage`
- `previous_marks`
- `assignments_completed`

The target/output column is:

- `final_exam_score`

In simple words, the model tries to learn how study behavior and previous academic performance influence the student's final marks.

## Why This Project Is Useful

This project shows how machine learning can be applied in education to estimate student performance. It can be useful for:

- identifying students who may need academic support
- understanding which factors influence marks the most
- demonstrating a regression-based ML workflow in a simple and understandable way

## Project Folder Structure

```text
aiml_sem2/
├── README.md
├── student_score_prediction/
│   ├── README.md
│   ├── VIVA_NOTES.md
│   ├── generate_dataset.py
│   ├── student_exam_scores.csv
│   └── student_score_prediction.py
└── other assignment PDFs
```

## Main Files and Their Purpose

### 1. `student_score_prediction/generate_dataset.py`

This file generates the dataset used in the project.

It creates student records containing:

- study hours
- attendance percentage
- previous marks
- assignments completed
- final exam score

Its purpose is to provide training data for the ML models.

### 2. `student_score_prediction/student_exam_scores.csv`

This is the dataset file used by the model.

It contains the student records in tabular format. Each row represents one student, and each column represents one feature or the final target score.

### 3. `student_score_prediction/student_score_prediction.py`

This is the main program of the project.

It performs the full machine learning workflow:

- loads the dataset
- analyzes the data
- checks for missing and duplicate values
- studies feature relationships
- splits data into training and testing sets
- trains multiple regression models
- evaluates them
- chooses the best model
- takes user input from the terminal
- predicts the final exam score

### 4. `student_score_prediction/VIVA_NOTES.md`

This file contains short viva-friendly explanations of the project.

## Technologies and Libraries Used

The project is built in Python using:

- `pandas` for data loading and analysis
- `scikit-learn` for machine learning models and evaluation
- `pathlib` for file path handling

The machine learning models used are:

- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor`

## Type of Machine Learning Problem

This project is a **supervised learning** problem because:

- we train the model using input data and known output values
- the model learns from examples where the final exam score is already available

More specifically, it is a **regression problem** because the output is a numeric value, not a category.

## Input Features and Their Meaning

### `hours_studied`

This represents how many hours a student studied.

Reason for choosing it:
Students who study more usually perform better in exams.

### `attendance_percentage`

This shows how regularly the student attended classes.

Reason for choosing it:
Attendance reflects classroom participation and exposure to teaching.

### `previous_marks`

This represents the student's earlier academic performance.

Reason for choosing it:
Students with good previous marks often have a stronger conceptual base.

### `assignments_completed`

This indicates how many assignments the student completed.

Reason for choosing it:
Assignments show practice, consistency, and revision effort.

### Target: `final_exam_score`

This is the value the model predicts.

## Complete Working of the Project

The project follows a standard machine learning pipeline.

### Step 1: Load the Dataset

The script loads the CSV file using `pandas.read_csv()`.

In the code:

- `BASE_DIR` stores the script directory
- `DATA_FILE` points to `student_exam_scores.csv`
- `load_data()` reads the dataset

Why this step is important:
The model needs organized historical data before it can learn patterns.

### Step 2: Student Data Analysis

The function `student_data_analysis(data)` performs basic data inspection.

It prints:

- first 5 rows of the dataset
- dataset shape
- descriptive statistics
- meaning of every feature

Why this step is important:
It helps us understand the data before model training.

Viva explanation:
We first inspect the data to know what values are present and what each column means.

### Step 3: Data Preparation

The function `data_preparation(data)` checks:

- missing values
- duplicate rows

Then it removes duplicates using `drop_duplicates()`.

Why this step is important:
If the data contains repeated or missing values, the model may learn incorrect patterns.

Viva explanation:
Clean data improves the reliability of the model.

### Step 4: Performance Insights

The function `performance_insights(data)` calculates correlation values between each feature and `final_exam_score`.

This helps us understand which features are more strongly related to the target.

Why this step is important:
It gives interpretability. We can explain why the model is making predictions.

Viva explanation:
This section shows how study hours, attendance, previous marks, and assignments affect final performance.

### Step 5: Feature and Target Separation

The function `prepare_features(data)` creates:

- `X` = input features
- `y` = target output

Selected input columns:

- `hours_studied`
- `attendance_percentage`
- `previous_marks`
- `assignments_completed`

Target column:

- `final_exam_score`

### Step 6: Train-Test Split

The same function uses `train_test_split()` to divide the data:

- `80%` for training
- `20%` for testing

Why this step is important:
The model should be evaluated on unseen data, not only on the data it has already seen.

Viva explanation:
Training data teaches the model. Testing data checks whether the model predicts well on new examples.

### Step 7: Model Training

The function `score_prediction_model(x_train, x_test, y_train, y_test)` trains three models:

#### Linear Regression

This model assumes a mostly direct relationship between input features and output score.

Usefulness:
It is simple, fast, and easy to explain.

#### Decision Tree Regressor

This model splits the data into rule-based branches.

Usefulness:
It can capture non-linear relationships.

#### Random Forest Regressor

This model combines many decision trees and averages their predictions.

Usefulness:
It is more powerful and often performs better on complex patterns.

## Model Evaluation

Each model is evaluated using the function `evaluate_model(...)`.

The evaluation metrics are:

### 1. MAE

MAE means Mean Absolute Error.

It shows the average difference between actual score and predicted score.

Interpretation:
Lower MAE means better prediction accuracy.

### 2. RMSE

RMSE means Root Mean Squared Error.

It gives higher importance to larger errors.

Interpretation:
Lower RMSE means the model is making fewer large mistakes.

### 3. R2 Score

R2 score measures how well the model explains the variation in the data.

Interpretation:

- closer to `1` means better performance
- higher R2 means the model fits the data better

## How the Best Model Is Selected

After evaluation, the program compares all model results and selects the one with the highest `R2 Score`.

In the current project run, `Linear Regression` performs best.

Simple explanation for viva:
The dataset follows a mostly direct pattern, so a simple regression model can predict the score effectively.

## Final Score Prediction for a New Student

After training, the script asks the user to enter a new student's details from the terminal.

The program currently asks for:

- `Hours studied`
- `Attendance percentage`
- `Previous marks`
- `Assignments completed`

The input is converted into a `pandas DataFrame`, and the best trained model predicts the final exam score.

This is the practical output of the whole project.

## Code Flow Summary

The execution flow of `student_score_prediction.py` is:

1. `load_data()`
2. `student_data_analysis(data)`
3. `data_preparation(data)`
4. `performance_insights(cleaned_data)`
5. `prepare_features(cleaned_data)`
6. `score_prediction_model(x_train, x_test, y_train, y_test)`
7. `final_score_estimation(best_model)`
8. `conclusion(best_model)`

This complete flow is controlled by the `main()` function.

## How to Run the Project

Open a terminal and run:

```bash
cd student_score_prediction
python3 generate_dataset.py
python3 student_score_prediction.py
```

When you run `student_score_prediction.py`, the script will:

- display dataset analysis
- show data cleaning information
- print feature correlations
- train and compare three models
- display model evaluation scores
- ask you to enter student details
- predict the final exam score
- print the final conclusion

## Example Terminal Input

When prompted, you can enter values like:

```text
Hours studied: 7.5
Attendance percentage: 90
Previous marks: 78
Assignments completed: 8
```

Then the script predicts the estimated final exam score for that student.

## Why Linear Regression Works Well Here

In this dataset, the target score generally increases when:

- study hours increase
- previous marks increase
- assignment completion increases
- attendance remains reasonably good

Because the relationship is fairly direct, Linear Regression fits well and gives strong performance.

## Strengths of This Project

- simple and practical problem statement
- easy-to-understand input features
- clean end-to-end ML pipeline
- multiple models compared instead of only one
- user can enter new data for live prediction
- suitable for academic presentation and viva

## Limitations of This Project

Like every ML project, this one also has some limitations:

- the dataset is small
- the data is synthetic/generated, not collected from a real institution
- only a few input features are used
- external factors like health, difficulty level, and family background are not included

These limitations are good to mention in a viva because they show critical understanding of the project.

## Possible Future Improvements

You can mention these as future scope:

- use a real student dataset
- include more academic and behavioral features
- build a web interface for entering input
- save the trained model using `joblib` or `pickle`
- add graphs and visualizations
- perform hyperparameter tuning

## Viva Questions and Ready Answers

### 1. What is the aim of your project?

The aim is to predict a student's final exam score using study hours, attendance, previous marks, and assignments completed.

### 2. What type of machine learning problem is this?

It is a supervised learning regression problem because the output is a continuous numeric value.

### 3. Why did you choose these input features?

Because they are simple academic factors that directly affect student performance and are easy to explain.

### 4. Why did you use multiple models?

To compare different algorithms and choose the model that gives the best prediction accuracy.

### 5. What evaluation metrics did you use?

I used MAE, RMSE, and R2 score.

### 6. Which model performed best?

Linear Regression performed best for this dataset.

### 7. Why did Linear Regression perform best?

Because the relationship between the features and final exam score is mostly direct and linear in this dataset.

### 8. Why do we split data into training and testing sets?

To check whether the model works well on unseen data and does not only memorize the training data.

### 9. What is the practical output of this project?

The practical output is the predicted final exam score for a new student entered by the user.

### 10. What are the limitations of your project?

The main limitations are small dataset size, generated data, and limited number of features.

## Short Viva Conclusion

This project successfully demonstrates a basic machine learning regression system for predicting student exam scores.

It starts from data loading and analysis, performs cleaning and feature study, trains multiple models, compares them using standard metrics, selects the best one, and finally predicts the score for a new student entered by the user.

Because the code is simple, logical, and practical, it is well suited for a viva presentation.
