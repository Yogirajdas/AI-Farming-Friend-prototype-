import pandas as pd # data analysis
import numpy as np # linear algebra

#import libraries for data visualization
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score

#Observing Data

crop = pd.read_csv('sample_data/Crop_recommendation.csv')
crop.head(25)

#Exploratory Data Analysis

crop.info()
crop.describe()
crop['label'].unique()
crop['label'].value_counts()

plt.figure(figsize=(19,17))
sns.pairplot(data, hue = "label")
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.distplot(crop['temperature'],color="red",bins=15,hist_kws={'alpha':0.5})
plt.subplot(1, 2, 2)
sns.distplot(crop['ph'],color="green",bins=15,hist_kws={'alpha':0.5})

sns.jointplot(x="rainfall",y="humidity",data=crop[(crop['temperature']<40) & (crop['rainfall']>40)],height=10,hue="label")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(30,15))
sns.boxplot(x='label',y='ph',data=crop)


crop_summary = pd.pivot_table(crop,index=['label'],aggfunc='mean')
crop_summary.head()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['N'],
    name='Nitrogen',
    marker_color='mediumvioletred'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['P'],
    name='Phosphorous',
    marker_color='springgreen'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['K'],
    name='Potash',
    marker_color='dodgerblue'
))

fig.update_layout(title="N-P-K values comparision between crops",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)

fig.show()

print("parameter based best crop")
print("---------------------------------")
print("Crops which requires very High Ratio of Nitrogen Content in Soil:", crop[crop['N'] > 120]['label'].unique())
print("Crops which requires very High Ratio of Phosphorous Content in Soil:", crop[crop['P'] > 100]['label'].unique())
print("Crops which requires very High Ratio of Potassium Content in Soil:", crop[crop['K'] > 200]['label'].unique())
print("Crops which requires very High Rainfall:", crop[crop['rainfall'] > 200]['label'].unique())
print("Crops which requires very Low Temperature :", crop[crop['temperature'] < 10]['label'].unique())
print("Crops which requires very High Temperature :", crop[crop['temperature'] > 40]['label'].unique())
print("Crops which requires very Low Humidity:", crop[crop['humidity'] < 20]['label'].unique())
print("Crops which requires very Low pH:", crop[crop['ph'] < 4]['label'].unique())
print("Crops which requires very High pH:", crop[crop['ph'] > 9]['label'].unique())



#Correlation Matrix

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(crop.corr(), annot=True,cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.title('Correlation between different features', fontsize = 15, c='black')
plt.show()

#Trying different models

features = crop[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = crop['label']

acc = []
model = []

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

predicted_values = knn.predict(x_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('K Nearest Neighbours')
print("KNN Accuracy is: ", x)

print(classification_report(y_test,predicted_values))

score = cross_val_score(knn,features,target,cv=5)
print('Cross validation score: ',score)


knn_train_accuracy = knn.score(x_train,y_train) 
print("knn_train_accuracy = ",knn.score(x_train,y_train)) #Print Train Accuracy

knn_test_accuracy = knn.score(x_test,y_test) 
print("knn_test_accuracy = ",knn.score(x_test,y_test)) #Print Test Accuracy

mean_acc = np.zeros(20)
for i in range(1,21):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhat= knn.predict(x_test)
    mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)

mean_acc

loc = np.arange(1,21,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,21), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()

#DecisionTreeclassifier

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DT.fit(x_train,y_train)

predicted_values = DT.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("Decision Tree's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))

score = cross_val_score(DT, features, target,cv=5)
print('Cross validation score: ',score)


dt_train_accuracy = DT.score(x_train,y_train)
print("Training accuracy = ",DT.score(x_train,y_train)) #Print Train Accuracy

dt_test_accuracy = DT.score(x_test,y_test)
print("Testing accuracy = ",DT.score(x_test,y_test)) #Print Test Accuracy

#Randomforestclassifier

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train,y_train)

predicted_values = RF.predict(x_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('RF')
print("Random Forest Accuracy is: ", x)

print(classification_report(y_test,predicted_values))

score = cross_val_score(RF,features,target,cv=5)
print('Cross validation score: ',score)


rf_train_accuracy = RF.score(x_train,y_train)
print("Training accuracy = ",RF.score(x_train,y_train)) #Print Train Accuracy

rf_test_accuracy = RF.score(x_test,y_test)
print("Testing accuracy = ",RF.score(x_test,y_test)) #Print Test Accuracy

#NaiveBayesclassifier

from sklearn.naive_bayes import GaussianNB
NaiveBayes = GaussianNB()

NaiveBayes.fit(x_train,y_train)

predicted_values = NaiveBayes.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes Accuracy is: ", x)

print(classification_report(y_test,predicted_values))

score = cross_val_score(NaiveBayes,features,target,cv=5)
print('Cross validation score: ',score)


nb_train_accuracy = NaiveBayes.score(x_train,y_train)
print("Training accuracy = ",NaiveBayes.score(x_train,y_train)) #Print Train Accuracy

nb_test_accuracy = NaiveBayes.score(x_test,y_test)
print("Testing accuracy = ",NaiveBayes.score(x_test,y_test)) #Print Test Accuracy

#XGBoostclassifier

import xgboost as xgb
XB = xgb.XGBClassifier()
XB.fit(x_train,y_train)

predicted_values = XB.predict(x_test)

x = metrics.accuracy_score(y_test, predicted_values);
acc.append(x)
model.append('XGBoost')
print("XGBoost Accuracy is: ", x)

print(classification_report(y_test,predicted_values))

score = cross_val_score(XB,features,target,cv=5)
print('Cross validation score: ',score)


XB_train_accuracy = XB.score(x_train,y_train)
print("Training accuracy = ",XB.score(x_train,y_train)) #Print Train Accuracy

XB_test_accuracy = XB.score(x_test,y_test)
print("Testing accuracy = ",XB.score(x_test,y_test)) #Print Test Accuracy

#Comparingaccuracymetric

plt.figure(figsize=[14,7],dpi = 100, facecolor='white')
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('ML Algorithms')
sns.barplot(x = acc,y = model,palette='viridis')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
