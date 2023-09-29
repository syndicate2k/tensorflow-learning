import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def convert_smoking(value):
    if value == 'current':
        return '2'
    elif value in ['never', 'No Info']:
        return '0'
    elif value in ['ever', 'former', 'not current']:
        return '1'

def convert_gender(value):
    if value == 'Female':
        return '1'
    else:
        return '0'

data = pd.read_csv('src/diabetes_prediction.csv')

data.describe().style.format("{:.2f}")

data = data.drop_duplicates()
data = data[data['gender'] != 'Other']
 
data['smoking_history'] = data['smoking_history'].apply(convert_smoking)
data['gender'] = data['gender'].apply(convert_gender)

X = data.loc[:, data.columns != 'diabetes']
y = data['diabetes']

model = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=25)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Точность прогноза:",(metrics.accuracy_score(y_test, y_pred)*100).__format__('.2f'),"%")
