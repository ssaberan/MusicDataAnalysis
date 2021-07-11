import pandas as pd
from sklearn.svm import SVR

frame_3 = pd.read_csv('Training_Data_3.csv')
cvframe_3 = pd.read_csv('CV_Data_3.csv')

X_3_major = frame_3.ix[:,(1,2,3,4,5,6)].values
X_3_gpa = frame_3.ix[:,(0,2,3,4,5,6)].values
X_3_often = frame_3.ix[:,(0,1,3,4,5,6)].values
X_3_effective = frame_3.ix[:,(0,1,2,4,5,6)].values
X_3_helps = frame_3.ix[:,(0,1,2,3,5,6)].values
X_3_genre = frame_3.ix[:,(0,1,2,3,4,6)].values
X_3_energy = frame_3.ix[:,(0,1,2,3,4,5)].values
y_3_major = frame_3.ix[:,(0)].values
y_3_gpa = frame_3.ix[:,(1)].values
y_3_often = frame_3.ix[:,(2)].values
y_3_effective = frame_3.ix[:,(3)].values
y_3_helps = frame_3.ix[:,(4)].values
y_3_genre = frame_3.ix[:,(5)].values
y_3_energy = frame_3.ix[:,(6)].values
cvX_3_major = cvframe_3.ix[:,(1,2,3,4,5,6)].values
cvX_3_gpa = cvframe_3.ix[:,(0,2,3,4,5,6)].values
cvX_3_often = cvframe_3.ix[:,(0,1,3,4,5,6)].values
cvX_3_effective = cvframe_3.ix[:,(0,1,2,4,5,6)].values
cvX_3_helps = cvframe_3.ix[:,(0,1,2,3,5,6)].values
cvX_3_genre = cvframe_3.ix[:,(0,1,2,3,4,6)].values
cvX_3_energy = cvframe_3.ix[:,(0,1,2,3,4,5)].values
cvy_3_major = cvframe_3.ix[:,(0)].values
cvy_3_gpa = cvframe_3.ix[:,(1)].values
cvy_3_often = cvframe_3.ix[:,(2)].values
cvy_3_effective = cvframe_3.ix[:,(3)].values
cvy_3_helps = cvframe_3.ix[:,(4)].values
cvy_3_genre = cvframe_3.ix[:,(5)].values
cvy_3_energy = cvframe_3.ix[:,(6)].values

import numpy as np
np.std(y_3_gpa)

def gpa_accuracy(predictions, actuals):
	count = 0
	index = 0
	for label in predictions:
		if abs(label - actuals[index]) < 0.2:
			count += 1
		index += 1
	return count / len(predictions)


def accuracy(predictions, actuals):
	count = 0
	index = 0
	for label in predictions:
		if abs(label - actuals[index]) < 0.5:
			count += 1
		index += 1
	return count / len(predictions)


svr_3_major = SVR()
svr_3_gpa = SVR()
svr_3_often = SVR()
svr_3_effective = SVR()
svr_3_helps = SVR()
svr_3_genre = SVR()
svr_3_energy = SVR()

svr_3_major.fit(X_3_major, y_3_major)
svr_3_gpa.fit(X_3_gpa, y_3_gpa)
svr_3_often.fit(X_3_often, y_3_often)
svr_3_effective.fit(X_3_effective, y_3_effective)
svr_3_helps.fit(X_3_helps, y_3_helps)
svr_3_genre.fit(X_3_genre, y_3_genre)
svr_3_energy.fit(X_3_energy, y_3_energy)

pred_3_major = svr_3_major.predict(cvX_3_major)
pred_3_gpa = svr_3_gpa.predict(cvX_3_gpa)
pred_3_often = svr_3_often.predict(cvX_3_often)
pred_3_effective = svr_3_effective.predict(cvX_3_effective)
pred_3_helps = svr_3_helps.predict(cvX_3_helps)
pred_3_genre = svr_3_genre.predict(cvX_3_genre)
pred_3_energy = svr_3_energy.predict(cvX_3_energy)

major_acc = accuracy(pred_3_major, cvy_3_major)
gpa_acc = gpa_accuracy(pred_3_gpa, cvy_3_gpa)
often_acc = accuracy(pred_3_often, cvy_3_often)
effective_acc = accuracy(pred_3_effective, cvy_3_effective)
helps_acc = accuracy(pred_3_helps, cvy_3_helps)
genre_acc = accuracy(pred_3_genre, cvy_3_genre)
energy_acc = accuracy(pred_3_energy, cvy_3_energy)

def mostCommon(data):
	maxCount = 0
	maxElem = 0
	currElem = 0
	while currElem < 7:
		currCount = 0
		for elem in data:
			if abs(currElem - elem) <= 0.1:
				currCount += 1
		if currCount >= maxCount:
			maxCount = currCount
			maxElem = currElem
		currElem += 0.1
	return maxElem


def worth(acc, mCom, data):
	count = 0
	for label in data:
		if abs(label - mCom) < 0.5:
			count += 1
	comAcc = count / len(data)
	return acc / comAcc


def gpa_worth(acc, mCom, data):
	count = 0
	for label in data:
		if abs(label - mCom) < 0.2:
			count += 1
	comAcc = count / len(data)
	return acc / comAcc


major_acc
worth(major_acc, mostCommon(y_3_major), y_3_major)

gpa_acc
gpa_worth(gpa_acc, mostCommon(y_3_gpa), y_3_gpa)

often_acc
worth(often_acc, mostCommon(y_3_often), y_3_often)

effective_acc
worth(effective_acc, mostCommon(y_3_effective), y_3_effective)

helps_acc
worth(helps_acc, mostCommon(y_3_helps), y_3_helps)

genre_acc
worth(genre_acc, mostCommon(y_3_genre), y_3_genre)

energy_acc
worth(energy_acc, mostCommon(y_3_energy), y_3_energy)

from scipy.stats.stats import pearsonr

frame_full = pd.read_csv('Full_Data.csv')

y_full_major = frame_full.ix[:,(0)].values
y_full_gpa = frame_full.ix[:,(1)].values
y_full_often = frame_full.ix[:,(2)].values
y_full_effective = frame_full.ix[:,(3)].values
y_full_helps = frame_full.ix[:,(4)].values
y_full_genre = frame_full.ix[:,(5)].values
y_full_energy = frame_full.ix[:,(6)].values

pearsonr(y_full_major, y_full_gpa)
pearsonr(y_full_major, y_full_often)
pearsonr(y_full_major, y_full_effective)
pearsonr(y_full_major, y_full_helps)
pearsonr(y_full_major, y_full_genre)
pearsonr(y_full_major, y_full_energy)

pearsonr(y_full_gpa, y_full_major)
pearsonr(y_full_gpa, y_full_often)
pearsonr(y_full_gpa, y_full_effective)
pearsonr(y_full_gpa, y_full_helps)
pearsonr(y_full_gpa, y_full_genre)
pearsonr(y_full_gpa, y_full_energy)

pearsonr(y_full_often, y_full_major)
pearsonr(y_full_often, y_full_gpa)
pearsonr(y_full_often, y_full_effective)
pearsonr(y_full_often, y_full_helps)
pearsonr(y_full_often, y_full_genre)
pearsonr(y_full_often, y_full_energy)

pearsonr(y_full_effective, y_full_major)
pearsonr(y_full_effective, y_full_gpa)
pearsonr(y_full_effective, y_full_often)
pearsonr(y_full_effective, y_full_helps)
pearsonr(y_full_effective, y_full_genre)
pearsonr(y_full_effective, y_full_energy)

pearsonr(y_full_helps, y_full_major)
pearsonr(y_full_helps, y_full_gpa)
pearsonr(y_full_helps, y_full_often)
pearsonr(y_full_helps, y_full_effective)
pearsonr(y_full_helps, y_full_genre)
pearsonr(y_full_helps, y_full_energy)

pearsonr(y_full_genre, y_full_major)
pearsonr(y_full_genre, y_full_gpa)
pearsonr(y_full_genre, y_full_often)
pearsonr(y_full_genre, y_full_effective)
pearsonr(y_full_genre, y_full_helps)
pearsonr(y_full_genre, y_full_energy)

pearsonr(y_full_energy, y_full_major)
pearsonr(y_full_energy, y_full_gpa)
pearsonr(y_full_energy, y_full_often)
pearsonr(y_full_energy, y_full_effective)
pearsonr(y_full_energy, y_full_helps)
pearsonr(y_full_energy, y_full_genre)

X_3_major_featureselected = frame_3.ix[:,(1,2,3,5)].values
X_3_gpa_featureselected = frame_3.ix[:,(0,2,3,5,6)].values
X_3_often_featureselected = frame_3.ix[:,(0,1,3,4,5,6)].values
X_3_effective_featureselected = frame_3.ix[:,(0,1,2,4,5,6)].values
X_3_helps_featureselected = frame_3.ix[:,(2,3)].values
X_3_genre_featureselected = frame_3.ix[:,(1,2,3,6)].values
X_3_energy_featureselected = frame_3.ix[:,(0,1,2,3,5)].values
cvX_3_major_featureselected = cvframe_3.ix[:,(1,2,3,5)].values
cvX_3_gpa_featureselected = cvframe_3.ix[:,(0,2,3,5,6)].values
cvX_3_often_featureselected = cvframe_3.ix[:,(0,1,3,4,5,6)].values
cvX_3_effective_featureselected = cvframe_3.ix[:,(0,1,2,4,5,6)].values
cvX_3_helps_featureselected = cvframe_3.ix[:,(2,3)].values
cvX_3_genre_featureselected = cvframe_3.ix[:,(1,2,3,6)].values
cvX_3_energy_featureselected = cvframe_3.ix[:,(0,1,2,3,5)].values

svr_3_major_featureselected = SVR()
svr_3_gpa_featureselected = SVR()
svr_3_often_featureselected = SVR()
svr_3_effective_featureselected = SVR()
svr_3_helps_featureselected = SVR()
svr_3_genre_featureselected = SVR()
svr_3_energy_featureselected = SVR()

svr_3_major_featureselected.fit(X_3_major_featureselected, y_3_major)
svr_3_gpa_featureselected.fit(X_3_gpa_featureselected, y_3_gpa)
svr_3_often_featureselected.fit(X_3_often_featureselected, y_3_often)
svr_3_effective_featureselected.fit(X_3_effective_featureselected, y_3_effective)
svr_3_helps_featureselected.fit(X_3_helps_featureselected, y_3_helps)
svr_3_genre_featureselected.fit(X_3_genre_featureselected, y_3_genre)
svr_3_energy_featureselected.fit(X_3_energy_featureselected, y_3_energy)

pred_3_major_featureselected = svr_3_major_featureselected.predict(cvX_3_major_featureselected)
pred_3_gpa_featureselected = svr_3_gpa_featureselected.predict(cvX_3_gpa_featureselected)
pred_3_often_featureselected = svr_3_often_featureselected.predict(cvX_3_often_featureselected)
pred_3_effective_featureselected = svr_3_effective_featureselected.predict(cvX_3_effective_featureselected)
pred_3_helps_featureselected = svr_3_helps_featureselected.predict(cvX_3_helps_featureselected)
pred_3_genre_featureselected = svr_3_genre_featureselected.predict(cvX_3_genre_featureselected)
pred_3_energy_featureselected = svr_3_energy_featureselected.predict(cvX_3_energy_featureselected)

major_acc_featureselected = accuracy(pred_3_major_featureselected, cvy_3_major)
gpa_acc_featureselected = gpa_accuracy(pred_3_gpa_featureselected, cvy_3_gpa)
often_acc_featureselected = accuracy(pred_3_often_featureselected, cvy_3_often)
effective_acc_featureselected = accuracy(pred_3_effective_featureselected, cvy_3_effective)
helps_acc_featureselected = accuracy(pred_3_helps_featureselected, cvy_3_helps)
genre_acc_featureselected = accuracy(pred_3_genre_featureselected, cvy_3_genre)
energy_acc_featureselected = accuracy(pred_3_energy_featureselected, cvy_3_energy)

major_acc_featureselected
worth(major_acc_featureselected, mostCommon(y_3_major), y_3_major)

gpa_acc_featureselected
gpa_worth(gpa_acc_featureselected, mostCommon(y_3_gpa), y_3_gpa)

often_acc_featureselected
worth(often_acc_featureselected, mostCommon(y_3_often), y_3_often)

effective_acc_featureselected
worth(effective_acc_featureselected, mostCommon(y_3_effective), y_3_effective)

helps_acc_featureselected
worth(helps_acc_featureselected, mostCommon(y_3_helps), y_3_helps)

genre_acc_featureselected
worth(genre_acc_featureselected, mostCommon(y_3_genre), y_3_genre)

energy_acc_featureselected
worth(energy_acc_featureselected, mostCommon(y_3_energy), y_3_energy)