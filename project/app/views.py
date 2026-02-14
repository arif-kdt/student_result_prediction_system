from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import joblib
import numpy as np
import os
from .ml_model import train_model 

def home(request):
    if not os.path.exists('student_model.pkl'):
        train_model()

    result = None

    if request.method == 'POST':
        marks = int(request.POST['marks'])
        attendance = int(request.POST['attendance'])
        assignment = int(request.POST['assignment'])

        model = joblib.load('student_model.pkl')
        prediction = model.predict([[marks, attendance, assignment]])

        result = "PASS ✅" if prediction[0] == 1 else "FAIL ❌"

    return render(request, 'index.html', {'result': result})
