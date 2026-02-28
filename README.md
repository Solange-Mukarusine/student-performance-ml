# Student Academic Performance Prediction (AI Project)

## Project Overview

This project applies **Supervised Machine Learning** using the **Random Forest Classification Algorithm** to predict students at academic risk.

The goal is to help institutions identify struggling students early and provide timely academic interventions.


## Project Developers

SOLANGE MUKARUSINE 

Course: Artificial Intelligence  
Project Type: Supervised Machine Learning  

##Machine Learning Model

Algorithm Used:
- Random Forest Classifier (Scikit-learn)

Target Variable:
- `final_result`
    - 1 → At Risk
    - 0 → Not At Risk

Features:
- Attendance
- Study Hours
- Continuous Assessment
- Participation Score
- Previous GPA


## Model Performance

- Accuracy: ~90%+
- Strong Recall for identifying at-risk students
- Feature importance analysis highlights attendance and assessment scores as key predictors

Visualizations included:
- Accuracy chart
- Confusion Matrix
- Feature Importance graph


## Project Structure
student-performance-ml/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── student_performance_model.ipynb
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── docs/
│   └── Final_Project_Report_Professional.pdf
│
├── app.py
├── requirements.txt
└── README.md
## How to Run the Project

### Clone the Repository

https://github.com/Solange-Mukarusine/student-performance-ml.git

## cd student-performance-ml

### Install Dependencies

pip install -r requirements.txt

### Run the Model Training Script (Optional)

If you want to retrain the model:

python src/train_model.py

###  Run the Flask Web Application

python app.py

Open your browser and go to:

http://127.0.0.1:5000

## Future Improvements

- Expand dataset with real university data
- Compare with SVM and Gradient Boosting
- Deploy to cloud (Render, Heroku, AWS)
- Build full dashboard interface


##License

This project is for academic purposes.
