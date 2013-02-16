import scipy
from sklearn import svm
from sklearn.svm import SVC

import csv

def get_precision_recall(labels, predictions):
	tp = fp = tn = fn = 0
	for i in range(len(labels)):
		label = labels[i]
		prediction = predictions[i]
		if label == 1:
			if prediction == 1:
				tp += 1
			else:
				fn += 1
		else:
			if prediction == 0:
				tn += 1
			else:
				fp += 1
	precision = float(tp) / (tp + fp)
	recall = float(tp) / (tp + fn)
	return [precision, recall]

def get_bad_words():
	badwords = set(word.strip().lower() for word in (open("../data/badwords.txt").readlines()))
	return badwords

def get_train_data():
	sentences = []
	labels = []
	reader = csv.reader(open("../data/kaggle/train.csv"), delimiter=',')
	reader.next()
	for row in reader:
		sentences.append(row[2])
		labels.append(int(row[0].strip()))
	return [sentences, labels]


def baseline(sentences):
	predictions = []
	badwords = get_bad_words()
	for line in sentences:
		words = set(line.lower().split())
		if not words & badwords:
			predictions.append(0)
		else:
			predictions.append(1)
	return predictions

def evaluate(predictions, labels):
	precision, recall = get_precision_recall(labels, predictions)
	print "precision: " + str(precision)
	print "recall: " + str(recall)

	F1 = (2.0 * precision * recall) / (precision + recall)

	print "F1: " + str(F1)

	
sentences, labels = get_train_data()
results = baseline(sentences)
evaluate(results, labels)

# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y) 
# print "hello"
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
# gamma=0.0, kernel='rbf', max_iter=-1, probability=False, shrinking=True,
# tol=0.001, verbose=False)


# print clf.predict([[2.,2.], [0, -1]])
