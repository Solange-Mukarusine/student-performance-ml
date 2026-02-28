from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Store prediction history (temporary memory storage)
prediction_history = []

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Student Risk Prediction</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        body.dark-mode {
            background-color: #121212;
            color: white;
        }
        .dark-mode .card {
            background-color: #1e1e1e;
            color: white;
        }
        .dark-mode table {
            color: white;
        }
    </style>
</head>

<body class="bg-light" id="body">

<div class="container mt-4">

    <div class="d-flex justify-content-end">
        <button onclick="toggleDarkMode()" class="btn btn-secondary btn-sm">
            Toggle Dark Mode
        </button>
    </div>

    <div class="card shadow-lg p-4 mt-3">
        <h2 class="text-center mb-4">Student Academic Risk Prediction</h2>

        <form method="POST">

            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Attendance (%)</label>
                    <input type="number" name="attendance" class="form-control"
                           min="0" max="100" required>
                </div>

                <div class="col-md-6 mb-3">
                    <label class="form-label">Study Hours</label>
                    <input type="number" name="study_hours" class="form-control"
                           min="0" max="80" required>
                </div>

                <div class="col-md-6 mb-3">
                    <label class="form-label">Continuous Assessment</label>
                    <input type="number" name="continuous_assessment" class="form-control"
                           min="0" max="100" required>
                </div>

                <div class="col-md-6 mb-3">
                    <label class="form-label">Participation (1–10)</label>
                    <input type="number" name="participation_score" class="form-control"
                           min="1" max="10" required>
                </div>

                <div class="col-md-6 mb-3">
                    <label class="form-label">Previous GPA (0–4.0)</label>
                    <input type="number" name="previous_gpa" class="form-control"
                           min="0" max="4" step="0.01" required>
                </div>
            </div>

            <button type="submit" class="btn btn-primary w-100">Predict</button>
        </form>

        {% if prediction %}
        <div class="alert alert-info mt-4 text-center">
            <h4>{{ prediction }}</h4>
            <p><strong>Risk Probability:</strong> {{ probability }}%</p>
        </div>
        {% endif %}

        {% if error %}
        <div class="alert alert-danger mt-4 text-center">
            {{ error }}
        </div>
        {% endif %}

    </div>

    {% if history %}
    <div class="card mt-4 p-3 shadow">
        <h4>Prediction History</h4>
        <table class="table table-bordered table-striped mt-2">
            <thead>
                <tr>
                    <th>Attendance</th>
                    <th>Study Hours</th>
                    <th>GPA</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                </tr>
            </thead>
            <tbody>
                {% for row in history %}
                <tr>
                    <td>{{ row.attendance }}</td>
                    <td>{{ row.study_hours }}</td>
                    <td>{{ row.gpa }}</td>
                    <td>{{ row.prediction }}</td>
                    <td>{{ row.probability }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

</div>

<script>
function toggleDarkMode() {
    document.getElementById("body").classList.toggle("dark-mode");
}
</script>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            attendance = float(request.form["attendance"])
            study_hours = float(request.form["study_hours"])
            continuous_assessment = float(request.form["continuous_assessment"])
            participation_score = float(request.form["participation_score"])
            previous_gpa = float(request.form["previous_gpa"])

            # Validation
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be 0–100.")
            if not (0 <= previous_gpa <= 4):
                raise ValueError("GPA must be 0–4.0.")

            values = np.array([[attendance, study_hours,
                                continuous_assessment,
                                participation_score,
                                previous_gpa]])

            values_scaled = scaler.transform(values)

            result = model.predict(values_scaled)[0]
            prob = model.predict_proba(values_scaled)[0][1] * 100

            probability = round(prob, 2)

            if result == 1:
                prediction = "Student is At Risk"
            else:
                prediction = "Student is Not At Risk"

            # Save to history
            prediction_history.append({
                "attendance": attendance,
                "study_hours": study_hours,
                "gpa": previous_gpa,
                "prediction": prediction,
                "probability": probability
            })

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        prediction=prediction,
        probability=probability,
        error=error,
        history=prediction_history
    )

if __name__ == "__main__":
    app.run(debug=True)
