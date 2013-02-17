import scipy
from sklearn import svm
from sklearn.svm import SVC
import numpy

import csv

class Feature(object):
	def __init__(self):
		self.me = "hello"
		pass


	def extract_all(self, sentences):
		all_values = []
		for sentence in sentences:
			values = self.extract(sentence)
			all_values.append(values)
		return all_values


class WordFeature(Feature):
	def __init__(self, words, is_binary=False):
		self._keywords = words
		self._is_binary = is_binary

	def extract(self, line):
		words = line.lower().split()
		count = 0.0
		for word in words:
			if word in self._keywords:
				count += 1
		if self._is_binary:
			return [1] if count > 0 else [0]
		else:
			return [10*(count / len(words))]
class WordPosition(Feature):
	def __init__(self, word, position=0):
		self._word = word
		self._position = position

	def extract(self, line):
		words = line.lower().split()
		if self._word in words:
			pos = words.index(self._word)
			return (1.0 / (1 + abs(self._position - pos)))
		return 0.0



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

def get_data(filename):
	sentences = []
	labels = []
	reader = csv.reader(open(filename), delimiter=',')
	reader.next()
	for row in reader:
		sentences.append(row[2])
		labels.append(int(row[0].strip()))
	return [sentences, labels]

def get_train_data():
	return get_data("../data/kaggle/train.csv")

def get_dev_data():
	return get_data("../data/kaggle/dev.csv")

def get_test_data():
	return get_data("../data/kaggle/test.csv")

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

def evaluate(predictions, labels, name):
	print "------Results for " + name + " model------------"
	precision, recall = get_precision_recall(labels, predictions)
	print "precision: " + str(precision)
	print "recall: " + str(recall)

	F1 = (2.0 * precision * recall) / (precision + recall)

	print "F1: " + str(F1)

def run_baseline():
	sentences, labels = get_dev_data()
	results = baseline(sentences)
	evaluate(results, labels, "baseline")

def get_feature_values(sentences):
	you_feature = WordFeature(["you", "u", "you're", "your"])
	values = you_feature.extract_all(sentences)

	swear_feature = WordFeature(get_bad_words(), True)
	swear_values = swear_feature.extract_all(sentences)

	pos_feature = WordPosition("you", 0)
	pos_values = pos_feature.extract_all(sentences)

	matrix = numpy.column_stack((values, swear_values, pos_values))
	return matrix


run_baseline()

train_sentences, train_labels = get_train_data()
matrix = get_feature_values(train_sentences)

print matrix[0]
#print train_labels
clf = svm.SVC()
clf.fit(matrix, train_labels) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, shrinking=True,
tol=0.001, verbose=False)


dev_sentences, dev_labels = get_dev_data()

matrix_test = get_feature_values(dev_sentences)

results = clf.predict(matrix_test)
#print results
evaluate(results, dev_labels, "SVM")
