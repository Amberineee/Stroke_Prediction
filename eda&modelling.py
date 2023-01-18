#libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_score,recall_score,f1_score

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

stroke_df[numerical_v].describe()
nonstroke_df[numerical_v].describe()

# stats of numerical data
round(stroke_df.describe(exclude = 'object'), 2)
round(nonstroke_df.describe(exclude = 'object'), 2)
# stats of categorical data
round(stroke_df.describe(exclude = ['float', 'int64']),2)
round(nonstroke_df.describe(exclude = ['float', 'int64']),2)

#hypertension, heart_disease, gender, ever_married, work_type, residence_type, smoking_status
sns.countplot(data = df, x = 'hypertension', hue='stroke')
plt.show()

sns.countplot(data = df, x = 'heart_disease', hue='stroke')
plt.show()

sns.countplot(data = df, x = 'gender', hue='stroke')
plt.show()

sns.countplot(data = df, x = 'ever_married', hue='stroke')
plt.show()

sns.countplot(data = df, x = 'work_type', hue='stroke')
plt.show()

sns.countplot(data = df, x = 'Residence_type', hue='stroke')
plt.show()

sns.countplot(data = df, x = 'smoking_status', hue='stroke')
plt.show()

#correlations between numerical variables
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()

#Stroke Rate Increases as Age Increases
df['age'] = df['age'].astype(int)
rate = []
for i in range(df['age'].min(), df['age'].max()):
    rate.append(df[df['age'] < i]['stroke'].sum() / len(df[df['age'] < i]['stroke']))

sns.lineplot(data=rate,color='#0f4c81')
plt.xlabel('age')
plt.ylabel('stroke rate')
plt.title('Stroke Rate Increases as Age Increases')
plt.show()

# binning of numerical variables
df['bmi_cat'] = pd.cut(df['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
df['age_cat'] = pd.cut(df['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
df['glucose_cat'] = pd.cut(df['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])

df.head(10)

#BMI_cat - Age - Avg_Glucose - Stroke
g = sns.FacetGrid(df, col="bmi_cat", hue='stroke', 
                  col_order=['Underweight', 'Ideal', 'Overweight', 'Obesity'], hue_order=[0,1],
                  aspect=1.2, size=3.5, palette=sns.light_palette('brown', 2))
g.map(plt.scatter, "age", "avg_glucose_level", alpha=0.9, 
      edgecolor='white', linewidth=0.5)
    
fig = g.fig 
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('BMI_cat - Age - Avg_Glucose - Stroke', fontsize=14)
l = g.add_legend(title='Stroke')
plt.show()

#Age_cat - BMI - Avg_Glucose - Stroke
g = sns.FacetGrid(df, col="age_cat", hue='stroke', 
                  col_order=['Children', 'Teens', 'Adults','Mid Adults','Elderly'], hue_order=[0,1],
                  aspect=1.2, size=3.5, palette=sns.light_palette('brown', 2))
g.map(plt.scatter, "bmi", "avg_glucose_level", alpha=0.9, 
      edgecolor='white', linewidth=0.5)
    
fig = g.fig 
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Age_cat - BMI - Avg_Glucose - Stroke', fontsize=14)
l = g.add_legend(title='Stroke')
plt.show()

#Avg_Glucose_cat - Age - BMI - Stroke
g = sns.FacetGrid(df, col="glucose_cat", hue='stroke', 
                  col_order=['Low', 'Normal', 'High', 'Very High'], hue_order=[0,1],
                  aspect=1.2, size=3.5, palette=sns.light_palette('brown', 2))
g.map(plt.scatter, "age", "bmi", alpha=0.9, 
      edgecolor='white', linewidth=0.5)
    
fig = g.fig 
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Avg_Glucose_cat - Age - BMI - Stroke', fontsize=14)
l = g.add_legend(title='Stroke')
plt.show()

#Machine Learning Modelling
# create encoder for each categorical variable
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
label_gender = LabelEncoder()
label_married = LabelEncoder()
label_work = LabelEncoder()
label_residence = LabelEncoder()
label_smoking = LabelEncoder()

df['gender'] = label_gender.fit_transform(df['gender'])
df['ever_married'] = label_married.fit_transform(df['ever_married'])
df['work_type']= label_work.fit_transform(df['work_type'])
df['Residence_type']= label_residence.fit_transform(df['Residence_type'])
df['smoking_status']= label_smoking.fit_transform(df['smoking_status'])

# fit the object to our training data
smote = SMOTE(sampling_strategy='minority')
X,y = df[['gender','age','hypertension','heart_disease','ever_married', 'work_type','Residence_type', 'avg_glucose_level','bmi', 'smoking_status']], df['stroke']
X, y = smote.fit_resample(X,y)
usampled_df = X.assign(Stroke = y)
print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))

_, class_counts = np.unique(y, return_counts=True)
class_names = ['No stroke', 'Stroke']
fig, ax = plt.subplots()
ax.pie(class_counts, labels=class_names, autopct='%1.2f%%',
      startangle=90, counterclock=False)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title('Class distribution')
plt.show()
print("# samples associated with no stroke: {}".format(class_counts[0]))
print("# samples associated with stroke: {}".format(class_counts[1]))

#data splitting
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 890)

# Null accuracy Score for current data
NUll_acc = round(max(y_test.mean(), 1 - y_test.mean()), 2)
print('Null Accuracy Score for Current Data is {}'.format(NUll_acc))

#data normalization 
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#model training and evaluation 
#Singular Vector Machine (SVM)
svm = SVC(kernel='rbf',probability=True)
svm.fit(X_train_std, y_train)

y_svm = svm.predict(X_test_std)
y_svm_prob = svm.predict_proba(X_test_std)

print("Classification report for SVM: \n{}".format(classification_report(y_test,y_svm)))
print("Confusion matrix for SVM: \n{}".format(confusion_matrix(y_test,y_svm)))
print("Accuracy score for SVM: {:.2f}".format(accuracy_score(y_test,y_svm)))

#confusion matrix 
cm = confusion_matrix(y_test,y_svm)
class_names=['non-stroke','stroke'] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# calculate precision, recall, and f1 scores
prec_svm = precision_score(y_test,y_svm)
rec_svm = recall_score(y_test,y_svm)
f1_svm = f1_score(y_test,y_svm)
print("Precision score for SVM: {:.2f}".format(prec_svm))
print("Recall score for SVM: {:.2f}".format(rec_svm))
print("F1 score for SVM: {:.2f}".format(f1_svm))

fpr, tpr, _ = roc_curve(y_test,  y_svm_prob[:,1])
auc_svm = roc_auc_score(y_test, y_svm_prob[:,1])
print("AUC score for SVM: {:.2f}".format(auc_svm))

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_svm)
ax.plot([0, 1], [0, 1], color='green', linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (SVM)')
ax.legend(loc="lower right")
plt.show()

#Logistic Regression (LR)
logistic = LogisticRegression(random_state=890)
logistic.fit(X_train_std, y_train)

y_logit = logistic.predict(X_test_std)
y_logit_prob = logistic.predict_proba(X_test_std)

print("Classification report for Logistic Regression: \n{}".format(classification_report(y_test,y_logit)))
print("Confusion matrix for Logistic Regression: \n{}".format(confusion_matrix(y_test,y_logit)))
print("Accuracy score for Logistic Regression: {:.2f}".format(accuracy_score(y_test,y_logit)))

#confusion matrix 
cm = confusion_matrix(y_test,y_logit)
class_names=['non-stroke','stroke'] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# calculate precision, recall, and f1 scores
prec_lg = precision_score(y_test,y_logit)
rec_lg = recall_score(y_test,y_logit)
f1_lg = f1_score(y_test,y_logit)
print("Precision score for Logistic Regression: {:.2f}".format(prec_lg))
print("Recall score for Logistic Regression: {:.2f}".format(rec_lg))
print("F1 score for Logistic Regression: {:.2f}".format(f1_lg))

fpr, tpr, _ = roc_curve(y_test,  y_logit_prob[:,1])
auc_logistic = roc_auc_score(y_test, y_logit_prob[:,1])
print("AUC score for Logistic Regression: {:.2f}".format(auc_logistic))

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_logistic)
ax.plot([0, 1], [0, 1], color='green', linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (LR)')
ax.legend(loc="lower right")
plt.show()

#Random Forest (RF)
randomf = RandomForestClassifier(n_estimators=100, random_state=890)
randomf.fit(X_train_std, y_train)

y_randomf = randomf.predict(X_test_std)
y_randomf_prob = randomf.predict_proba(X_test_std)

print("Classification report for Random Forest: \n{}".format(classification_report(y_test,y_randomf)))
print("Confusion matrix for Random Forest: \n{}".format(confusion_matrix(y_test,y_randomf)))
print("Accuracy score for Random Forest: {:.2f}".format(accuracy_score(y_test,y_randomf)))

#confusion matrix 
cm = confusion_matrix(y_test,y_randomf)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# calculate precision, recall, and f1 scores
prec_rf = precision_score(y_test,y_randomf)
rec_rf = recall_score(y_test,y_randomf)
f1_rf = f1_score(y_test,y_randomf)
print("Precision score for Random Forest: {:.2f}".format(prec_rf))
print("Recall score for Random Forest: {:.2f}".format(rec_rf))
print("F1 score for Random Forest: {:.2f}".format(f1_rf))

fpr, tpr, _ = roc_curve(y_test,  y_randomf_prob[:,1])
auc_randomf = roc_auc_score(y_test, y_randomf_prob[:,1])
print("AUC score for Random Forest: {:.2f}".format(auc_randomf))

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_randomf)
ax.plot([0, 1], [0, 1], color='green', linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (RF)')
ax.legend(loc="lower right")
plt.show()

#XGboost (XGB)
xgb = XGBClassifier(objective="binary:logistic", random_state=890)
xgb.fit(X_train_std, y_train)

y_xgb = xgb.predict(X_test_std)
y_xgb_prob = xgb.predict_proba(X_test_std)

print("Classification report for Logistic Regression: \n{}".format(classification_report(y_test,y_xgb)))
print("Confusion matrix for Logistic Regression: \n{}".format(confusion_matrix(y_test,y_xgb)))
print("Accuracy score for Logistic Regression: {:.2f}".format(accuracy_score(y_test,y_xgb)))

#confusion matrix 
cm = confusion_matrix(y_test,y_xgb)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# calculate precision, recall, and f1 scores
prec_xgb = precision_score(y_test,y_xgb)
rec_xgb = recall_score(y_test,y_xgb)
f1_xgb = f1_score(y_test,y_xgb)
print("Precision score for Logistic Regression: {:.2f}".format(prec_xgb))
print("Recall score for Logistic Regression: {:.2f}".format(rec_xgb))
print("F1 score for Logistic Regression: {:.2f}".format(f1_xgb))

fpr, tpr, _ = roc_curve(y_test,  y_xgb_prob[:,1])
auc_xgb = roc_auc_score(y_test, y_xgb_prob[:,1])
print("AUC score for Logistic Regression: {:.2f}".format(auc_xgb))

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_xgb)
ax.plot([0, 1], [0, 1], color='green', linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (XGB)')
ax.legend(loc="lower right")
plt.show()

#feature importance
#random forest
feature_imp = pd.Series(randomf.feature_importances_, index = ['gender','age','hypertension','heart_disease','ever_married', 'work_type','Residence_type', 'avg_glucose_level','bmi', 'smoking_status']).sort_values(ascending=False)
feature_imp
#viz
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.rc('ytick',labelsize=7)
plt.show()

#xgboost
feature_imp = pd.Series(xgb.feature_importances_, index = ['gender','age','hypertension','heart_disease','ever_married', 'work_type','Residence_type', 'avg_glucose_level','bmi', 'smoking_status']).sort_values(ascending=False)
feature_imp
#viz
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.rc('ytick',labelsize=7)
plt.show() 



#test

#test1
