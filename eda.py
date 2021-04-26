#libraries
import pandas as pd

#read data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#missing values 
df.isnull().sum()

#filling missing values 
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df.isnull().sum()