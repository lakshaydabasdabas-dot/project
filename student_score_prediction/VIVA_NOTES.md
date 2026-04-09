# Viva Notes: Student Exam Score Prediction System

## Project Title

Student Exam Score Prediction System

## Project Objective

The aim of this project is to predict a student's `final_exam_score` using simple academic factors.

The input features are:

- `hours_studied`
- `attendance_percentage`
- `previous_marks`
- `assignments_completed`

The output is:

- `final_exam_score`

## 1. Student Data Analysis

In this section, we study the dataset before training the model.

We check:

- how many rows and columns are present
- what each column means
- the average, minimum, and maximum values

Simple explanation:
This step helps us understand the student data clearly before making predictions.

## 2. Data Preparation

In this section, we clean the data.

We check:

- missing values
- duplicate rows

Simple explanation:
If the data is not clean, the model may give wrong results.

## 3. Performance Insights

In this section, we study how each feature affects the final exam score.

Interpretation:

- More `hours_studied` generally increase marks.
- Better `attendance_percentage` usually improves performance because the student attends more classes.
- Higher `previous_marks` show that the student already has a good academic base.
- More `assignments_completed` usually means better practice and revision.

Simple explanation:
This section helps us explain why the model predicts a higher or lower score.

## 4. Score Prediction Model

We train three machine learning models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Simple explanation:
We compare different models to find which one predicts student exam scores more accurately.

## 5. Model Evaluation

We evaluate the models using:

- `MAE`
- `RMSE`
- `R2 Score`

Simple explanation:

- Lower `MAE` means smaller average error.
- Lower `RMSE` means better prediction quality.
- Higher `R2 Score` means the model fits the data better.

## 6. Final Score Estimation

After training, we give a new student's details to the best model.

Example:

- Hours studied = 7.5
- Attendance = 90%
- Previous marks = 78
- Assignments completed = 8

Predicted final exam score:

- Around `91.87`

Simple explanation:
This is the final practical output of the project.

## Conclusion

In this project, `Linear Regression` performed best.

Reason:
The dataset follows a mostly direct pattern. When study hours, attendance, previous marks, and assignment completion increase, the final exam score also generally increases.

So Linear Regression was the best choice because it handled this simple relationship well.
