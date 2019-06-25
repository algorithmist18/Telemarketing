#Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as ss 
from dython.nominal import conditional_entropy, correlation_ratio, theils_u, cramers_v
import statsmodels.stats.multicomp as multi

#Jobs vs Housing
def jobs_housing(df):

	jobs = df.job.unique()
	js = {}

	print('\nProportion of housing by jobs:\n')

	for job in jobs:
		data = df.groupby(df.job).get_group(job)['housing'].value_counts()
		js.update({job : 100*data['yes']/(data['yes']+data['no'])})

	for key in js:
		print('{} = {:.2f}{}'.format(key, js[key], '%'))
	
def jobs_tendencies(df):

	jobs = df.job.unique()
	js = {}

	print('\nProportion of customers subscription by jobs:\n')

	for job in jobs:
		data = df.groupby(df.job).get_group(job)['subscribed'].value_counts()
		js.update({job : 100*data['yes']/(data['yes']+data['no'])})

	for key in js:
		print('{} = {:.2f}{}'.format(key, js[key], '%'))
	
	plt.hist(js.keys(), width = 0.4)
	plt.hist(js.values(), width = 0.35)
	plt.show()

def duration_analysis(col1, col2):

	classes = df[col1].unique()
	durations = {}

	print('\nAverage {} by {}s:\n'.format(col2, col1))

	for job in classes:

		data = df.groupby(df[col1]).get_group(job)[col2].mean()
		durations.update({job : data})

	for key in durations:

		print('{} = {:.2f}\n'.format(key, durations[key]))

def categorical_categorical(df, col1, col2):

	corr = theils_u(df[col1], df[col2])
	print(corr)

#Anova test between two variables (col1 = categorical, col2 = continuous/discrete)
def anova_test(df):

	col1 = 'job'
	col2 = 'duration'

	duration_frame = df[[col1, col2]].copy()
	groups = duration_frame.groupby(col1)

	#Job-groups
	admin = groups.get_group('admin.')['duration']
	bluecollar = groups.get_group('blue-collar')['duration']
	student = groups.get_group('student')['duration']
	housemaid = groups.get_group('housemaid')['duration']
	services = groups.get_group('services')['duration']
	unemployed = groups.get_group('unemployed')['duration']
	entrepreneur = groups.get_group('entrepreneur')['duration']
	selfemployed = groups.get_group('self-employed')['duration']
	retired = groups.get_group('retired')['duration']

	print('Admin:\n')
	print(admin.head())

	F, p = ss.f_oneway(admin, bluecollar, student, housemaid, services, unemployed, entrepreneur, selfemployed, retired)

	print('{} {}\n'.format(F, p))

	if p < 0.05:

		print('Null hypothesis rejected. Statistical difference found. Conduct post-hoc tests.')

	else:

		print('Not much difference found. Can accept null hypothesis.')

	#Conducting a Post-Hoc test
	duration_frame[col2] = duration_frame[col2].convert_objects(convert_numeric = True)

	mc = multi.MultiComparison(duration_frame[col2], duration_frame[col1])
	result = mc.tukeyhsd()
	print(result.summary())

#Reading data

df = pd.read_csv('train.csv')
df.drop('ID', axis = 1, inplace = True)
print(df.head())
#Analyzing data

#jobs_tendencies(df)
#jobs_housing(df)

#How long do people talk according to their jobs? 

duration_analysis('job', 'duration')
#duration_analysis('job', 'balance')

#Finding co-relations between variables and subscription result
"""
for col in df.columns:
	if df[col].dtype != object:
		print('{} vs Subscribed'.format(col))
		corr = correlation_ratio(df['subscribed'], df[col])
		print('{:.3f}'.format(corr))

print('\n\n\n')

for col in df.columns:
	if df[col].dtype == object:
		print('{} vs Subscribed'.format(col))
		corr = theils_u(df['subscribed'], df[col])
		print('{:.3f}'.format(corr))

"""
#Marital status and Loans: is there a relation?


print('\nAnalyzing marital status vs Loans:\n')
status = df.marital.unique()

for s in status:
	print(s)
	data = df.groupby(df.marital).get_group(s)['loan'].value_counts()
	print('{:.2f}{}'.format(100*data['yes']/(data['yes'] + data['no']), '%'))

corelation = theils_u(df['loan'], df['marital'])

print('\nCorrelation ratio = {:.3f}'.format(corelation))
print('Marital status does not influence loans')

#Jobs and Durations: again
anova_test(df)


