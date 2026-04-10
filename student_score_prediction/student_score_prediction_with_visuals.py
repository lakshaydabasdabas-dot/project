from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Import visualization module
import visualizations

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_exam_scores.csv"


def print_heading(title):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def load_data():
    return pd.read_csv(DATA_FILE)


def student_data_analysis(data):
    print_heading("Student Data Analysis")
    print("This section helps us understand the student dataset before training any model.")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nDataset shape:")
    print(data.shape)
    print("\nColumn summary:")
    print(data.describe())
    print("\nSimple meaning of each feature:")
    print("- hours_studied: More study time usually helps a student prepare better.")
    print("- attendance_percentage: Regular class attendance helps students follow lessons.")
    print("- previous_marks: Past performance gives an idea about the student's base level.")
    print("- assignments_completed: Finishing assignments shows regular practice.")
    print("- final_exam_score: This is the score we want to predict.")
    
    # Run basic visualizations
    input("\nPress Enter to see data visualizations...")
    visualizations.plot_correlation_heatmap(data)
    visualizations.plot_feature_vs_target(data)
    visualizations.plot_distribution(data)


def data_preparation(data):
    print_heading("Data Preparation")
    print("Here we check for missing values and make sure the dataset is ready.")
    print("\nMissing values in each column:")
    print(data.isnull().sum())
    print("\nDuplicate rows:")
    print(data.duplicated().sum())
    print("\nWhy this matters:")
    print("- Clean data gives more reliable predictions.")
    print("- If values are missing or repeated too much, the model can learn the wrong pattern.")
    return data.drop_duplicates()


def performance_insights(data):
    print_heading("Performance Insights")
    correlation = data.corr(numeric_only=True)["final_exam_score"].sort_values(ascending=False)
    print("This section shows how each feature is connected to the final exam score.")
    print("\nCorrelation with final exam score:")
    print(correlation)
    print("\nEasy interpretation:")
    print("- More study hours generally increase marks.")
    print("- Better attendance usually improves understanding and exam performance.")
    print("- Higher previous marks suggest the student already has a strong academic base.")
    print("- Completing more assignments usually means more revision and practice.")


def prepare_features(data):
    feature_columns = [
        "hours_studied",
        "attendance_percentage",
        "previous_marks",
        "assignments_completed",
    ]
    x = data[feature_columns]
    y = data["final_exam_score"]
    return train_test_split(x, y, test_size=0.2, random_state=42), feature_columns


def evaluate_model(model_name, model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    return {
        "name": model_name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "y_test": y_test
    }


def score_prediction_model(x_train, x_test, y_train, y_test, feature_names):
    print_heading("Score Prediction Model")
    print("We train three models and compare their performance.")
    print("1. Linear Regression: Best when the relationship is direct and simple.")
    print("2. Decision Tree: Splits data into rule-based branches.")
    print("3. Random Forest: Combines many decision trees for a stronger prediction.")

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42, max_depth=5)),
        ("Random Forest", RandomForestRegressor(random_state=42, n_estimators=200, max_depth=6)),
    ]

    results = []
    for model_name, model in models:
        model.fit(x_train, y_train)
        results.append(evaluate_model(model_name, model, x_test, y_test))
    
    # Show model comparison visualization
    input("\nPress Enter to see model comparison charts...")
    visualizations.plot_model_comparison(results)
    
    # Show feature importance for tree-based models
    for result in results:
        if result["name"] in ["Decision Tree", "Random Forest"]:
            input(f"\nPress Enter to see feature importance for {result['name']}...")
            visualizations.plot_feature_importance(result["model"], feature_names)
    
    # Show residual plots
    for result in results:
        input(f"\nPress Enter to see residual analysis for {result['name']}...")
        visualizations.plot_residuals(result["y_test"], result["predictions"], result["name"])
    
    best_model = max(results, key=lambda result: result["r2"])
    return best_model, results


def get_student_input():
    print("Enter student details for prediction:")
    hours_studied = float(input("Hours studied: "))
    attendance_percentage = float(input("Attendance percentage: "))
    previous_marks = float(input("Previous marks: "))
    assignments_completed = float(input("Assignments completed: "))

    return pd.DataFrame(
        [
            {
                "hours_studied": hours_studied,
                "attendance_percentage": attendance_percentage,
                "previous_marks": previous_marks,
                "assignments_completed": assignments_completed,
            }
        ]
    )


def final_score_estimation(best_model):
    print_heading("Final Score Estimation")
    sample_student = get_student_input()

    predicted_score = best_model["model"].predict(sample_student)[0]
    print("\nStudent input:")
    print(sample_student)
    print(f"\nEstimated final exam score: {predicted_score:.2f}")
    print("\nReal-life explanation:")
    print("The model estimated this score using study hours, attendance, previous marks,")
    print("and assignment completion as the main factors.")


def conclusion(best_model):
    print_heading("Conclusion")
    print(f"The best model is: {best_model['name']}")
    print(
        "Reason: It gave the most accurate prediction on test data based on the R2 score "
        "and error values."
    )
    if best_model["name"] == "Linear Regression":
        print(
            "This makes sense because student marks in this dataset follow a mostly direct pattern. "
            "As study hours, attendance, previous marks, and assignments increase, the final score also increases."
        )
    else:
        print(
            "This model performed better because it captured the score pattern more effectively than the other models."
        )


def main():
    print("Student Exam Score Prediction System with Visualizations")
    print("=" * 60)
    
    data = load_data()
    student_data_analysis(data)
    cleaned_data = data_preparation(data)
    performance_insights(cleaned_data)
    (x_train, x_test, y_train, y_test), feature_names = prepare_features(cleaned_data)
    best_model, _ = score_prediction_model(x_train, x_test, y_train, y_test, feature_names)
    final_score_estimation(best_model)
    conclusion(best_model)


if __name__ == "__main__":
    main()