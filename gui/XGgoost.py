# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib


df=pd.read_csv(r'D:\course\Courses\NTI final project\eda_file.csv')

cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity',
        'is_ph_neutral', 'is_hard', 'carbon_ratio', 'sulfate_to_solids',
        'ph_x_chloramines', 'hardness_minus_ph']

X = df[cols]
y = df["Potability"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = XGBClassifier(max_depth=4, learning_rate=0.1, random_state=42, n_estimators=7)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(f"acuuuracy for train model is {model.score(X_train,y_train)}")
print(f"acuuuracy for test model is {model.score(X_test,y_test)}")
y_pred=model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(y_proba[:100])
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
cals_report=classification_report(y_test,y_pred)
accuracyModel1=accuracy_score(y_test,y_pred)
print(f'Accuracy Score: {accuracyModel1:.2f}')

print(f'classification report:{cals_report} ')

cm = confusion_matrix(y_test, y_pred)
print(cm)

model.save_model("model.cmb")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model & Scaler saved successfully!")
