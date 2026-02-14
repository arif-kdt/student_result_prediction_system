import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    data = pd.read_csv('dataset.csv')

    X = data[['marks', 'attendance', 'assignment']]
    y = data['result']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, 'student_model.pkl')
