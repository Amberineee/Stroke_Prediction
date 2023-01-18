#libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#missing values 
df.isnull().sum()

#filling missing values 
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df.isnull().sum()

#stroke distribution 
pie, ax = plt.subplots(figsize=[10,6])
labels = ['Non-Stroke', 'Stroke']
plt.pie(x = df['stroke'].value_counts(), autopct="%.1f%%", labels = labels)
plt.title('Stroke Distribution')
plt.show()

#age, avg_glucose_level, bmi distribution
stroke_df = df.loc[df['stroke'] == 1]
numerical_v = ['age', 'avg_glucose_level', 'bmi']
stroke_df[numerical_v].hist(figsize=(20, 10), layout=(2, 4))
plt.show()

nonstroke_df = df.loc[df['stroke'] == 0]
nonstroke_df[numerical_v].hist(figsize=(20, 10), layout=(2, 4))
plt.show()



#test
