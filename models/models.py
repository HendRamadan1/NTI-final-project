
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PowerTransformer






class ModelTrainer:
   def __init__(self,df,target_column='Potability'):
       self.df=df
       self.target_column=target_column
       self.models={
           'Logistic Regression':LogisticRegression(),
           'Decision Tree':DecisionTreeClassifier(max_depth=8,max_features=20),
           'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=20,max_features=20),
           'XGBOOst':XGBClassifier(max_depth=10,learning_rate=0.2,random_state=42,n_estimators=10),
           "catBoosting":CatBoostClassifier(iterations=300,learning_rate=0.03,depth=8,eval_metric="Accuracy",verbose=50,random_state=0)   
       }
       
       
       
   def split_data(self, test_size=0.2, random_state=42):
        x = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        pt = PowerTransformer(method='yeo-johnson')
        x=pt.fit_transform(x)
        
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        print(x)
        
        return train_test_split(x, y, test_size=test_size, random_state=random_state)
   def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.split_data()
        result={}
        for name ,model in self.models.items():
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            acc=accuracy_score(y_test,y_pred)
            roc = roc_auc_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            result[name] = {
                "Accuracy": round(acc, 3),
                "ROC AUC": round(roc, 3),
                "F1-score": round(report['weighted avg']['f1-score'], 3)
            }

        return result


