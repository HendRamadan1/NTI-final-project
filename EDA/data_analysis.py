import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

df=pd.read_csv(r'D:\course\Courses\NTI final project\NTI-final-project-\data\row\water_potability.csv')
print(df)


##check duplicated 
print(df.duplicated().sum())
##check misiing values 
def missing_value1(df):
    
 missing_value=df.isnull().sum().sort_values(ascending=False)

 missing_value=missing_value[missing_value>0]
 missing_value=missing_value/len(df)
 print( missing_value)
 print(f"number of missing value is {len(missing_value)}")
 
missing_value1(df)

#handle miisng values 
df['ph']=df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('median'))
df['Sulfate']=df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('median'))
df['Trihalomethanes']=df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('median'))


###!handle outlier 
q1=df.quantile(0.25)
q3=df.quantile(0.75)
IQR=q3-q1
lower=q1-IQR*1.5
upper=q3+1.5*IQR
for x in df.columns:
    df[x]=np.where(df[x]>upper[x],upper[x],df[x])
    df[x]=np.where(df[x]<lower[x],lower[x],df[x])
    
    
###!add new feature 
df['is_ph_neutral']=df['ph'].between(6.5,8.5)
df['is_hard']=df["Hardness"]>300
df['carbon_ratio'] = df['Organic_carbon'] / (df['Solids'] + 1) 
df['sulfate_to_solids'] = df['Sulfate'] / (df['Solids'] + 1)
df['ph_x_chloramines'] = df['ph'] * df['Chloramines']
df['hardness_minus_ph'] = df['Hardness'] - df['ph']
df.to_csv("eda_file.csv")


    
