import csv
import random


OUTPUT_FILE = "student_exam_scores.csv"
ROW_COUNT = 120
SEED = 42


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


def main():
    random.seed(SEED)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "hours_studied",
                "attendance_percentage",
                "previous_marks",
                "assignments_completed",
                "final_exam_score",
            ]
        )

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

            writer.writerow(
                [
                    hours_studied,
                    attendance_percentage,
                    previous_marks,
                    assignments_completed,
                    final_exam_score,
                ]
            )


if __name__ == "__main__":
    main()
