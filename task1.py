# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Titanic-Dataset.csv")  

print(df.head())
print(df.info())
print(df.describe())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


sns.boxplot(x=df['Age'])
plt.title("Boxplot - Age")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Boxplot - Fare")
plt.show()

for col in ['Age', 'Fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("✅ Cleaned Data Preview:")
print(df.head())
