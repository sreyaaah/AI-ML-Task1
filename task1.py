import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Titanic-Dataset.csv")  

print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nFirst 5 Rows:\n", df.head())

print("Missing values before:", df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)  
df.fillna(df.mode().iloc[0], inplace=True)           
print("Missing values after:", df.isnull().sum())

categorical_cols = df.select_dtypes(include=["object", "category"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Categorical columns encoded:", [col for col in df.columns if 'Sex_' in col or 'Embarked_' in col])


numeric_cols = df.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df[numeric_cols].describe())


for col in numeric_cols:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

print("Before outlier removal:", df.shape)
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
          (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("After outlier removal:", df.shape)

print("\nFinal cleaned data shape:", df.shape)
