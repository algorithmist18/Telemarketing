#Importing libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm
import csv


def make_prediction(X_train, y_train, X_valid, y_valid):

	models = {}

	#Decision Tree Classifier: Maximum accuracy achieved at depth 6.
	print('Using Decision Tree classifier:\n')

	depths = [3, 4, 5, 6, 7, 8]
	results = []
	mx = 0.0
	
	for depth in depths:

		clf = tree.DecisionTreeClassifier(max_depth = depth, criterion = 'gini')
		clf = clf.fit(X_train, y_train)
		y_pred = clf.predict(X_valid)
		result = accuracy_score(y_pred, y_valid)
		results.append(result)

		if result > mx:
			mx = result
			models.update({'Decision Tree' : clf})

		#print('Accuracy = {:.2f}{}'.format(result*100, '%'))
		
	plt.plot(depths, results)
	plt.xlabel('Depth of Decision Tree')
	plt.ylabel('Accuracy')
	plt.show()

	print('\nDepth of 6 resulted in highest accuracy {:.2f}{} for Decision Tree classifier.\n'.format(mx*100, '%'))
	#Random Forest Classifier

	print('Using Random Forest Classifier:\n')

	depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
	results = []
	m = -1
	d = 0

	for depth in depths:

		clf = RandomForestClassifier(max_depth = depth)
		clf = clf.fit(X_train, y_train)
		y_pred = clf.predict(X_valid)
		result = accuracy_score(y_pred, y_valid)
		results.append(result)
		#print('Accuracy = {:.2f}{}'.format(result*100, '%'))
		if result > m:
			m = result
			d = depth
			models.update({'Random Forest' : clf})

	plt.plot(depths, results)
	plt.xlabel('Depth of Random Forest Classifier')
	plt.ylabel('Accuracy')
	plt.show()

	print('\nMaximum accuracy = {:.2f}{} for depth = {}\n'.format(m*100, '%', d))

	#Support Vector Machines classifier

	print('\nUsing SVM classifier:\n')

	clf = svm.SVC(kernel = 'sigmoid')
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_valid)
	result = accuracy_score(y_pred, y_valid)
	print('Accuracy = {:.2f}{}'.format(result*100, '%'))

	models.update({'SVM' : clf})

	return models

#Reading data

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df.drop(['ID'], axis = 1, inplace = True)
df_test.drop(['ID'], axis = 1, inplace = True)
#Clean training data

for col in df.columns:

	if df[col].dtype == object:
		#Get dummies (OHE)
		if col != 'subscribed':
			df = pd.concat([df, pd.get_dummies(df[col])], axis = 1)
			df.drop([col], axis = 1, inplace = True)
	"""
	else:
		#Scale data to unit variance
		df[col] -= df[col].min()
		df[col] /= df[col].max()
	"""
#Cleaning test data

for col in df_test.columns:

	if df_test[col].dtype == object:
		#Get dummies (OHE)
		if col != 'subscribed':
			df_test = pd.concat([df_test, pd.get_dummies(df_test[col])], axis = 1)
			df_test.drop([col], axis = 1, inplace = True)
	"""
	else:
		#Scale data to unit variance
		df_test[col] -= df_test[col].min()
		df_test[col] /= df_test[col].max()
	"""

X = df.drop(['subscribed'], axis = 1)
y = df['subscribed']

print(X.head())
print(y.head())

#Separate into training and validation set

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, random_state = 0)

print('\nAnalyzing validation data:\n')

#List of models to choose from

models = make_prediction(X_train, y_train, X_valid, y_valid)
mx = 0

#Checking which classifier has the highest score:
for key in models:

	value = models[key]
	y_pred = value.predict(X_valid)
	result = accuracy_score(y_pred, y_valid)

	print('\n{} = {:.2f}{}'.format(key, result*100, '%'))
	if result > mx:
		
		max_model = value
		model_name = key
		mx = result
 
y_test = max_model.predict(df_test)
print('\nUsing {} to predict test data:\n'.format(model_name))
print(y_test) 

with open('answer.txt', 'w') as f:
	for elem in y_test:
		f.write(elem)
		f.write('\n')

f.close()
print('\nFile updated succesfully.')