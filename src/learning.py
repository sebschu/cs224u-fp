import scipy
from sklearn import svm
from sklearn.svm import SVC
import numpy
import nltk
#from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re

class Feature(object):
	tokens_cache = dict()
	def __init__(self):
		pass

	def preprocess(self, sentence):
		sentence = re.sub(r'"', "", sentence)
		sentence = re.sub(r'(\w)(\1\1+)', r'\1', sentence)
		sentence = re.sub(r'_', " ", sentence)
		sentence = re.sub(r'@', "a", sentence)
		#sentence = re.sub(r'\$', "s", sentence)
		sentence = re.sub(r'\\+[ux][0-9a-f]+', " ", sentence)
		sentence = re.sub(r'\\+[nt]',' ',sentence)
		sentence = re.sub(r'\\+[\']','\'',sentence)
		sentence = re.sub(r'&\w+;',' ',sentence)
		return sentence

	def preprocess_all(self, sentences):
		new_sentences = []
		for sentence in sentences:
			new_sentences.append(self.preprocess(sentence))
		return new_sentences

	def tokenize(self, sentence):
		#print sentence
		sentence = self.preprocess(sentence)
		if sentence not in Feature.tokens_cache:
			tokenizer = TreebankWordTokenizer()
			Feature.tokens_cache[sentence] = tokenizer.tokenize(sentence)
		return Feature.tokens_cache[sentence]

	def extract_all(self, sentences):
		all_values = []
		for sentence in sentences:
			#print sentence + "\n\n"
			values = self.extract(sentence)
			all_values.append(values)
		return scipy.sparse.coo_matrix(all_values)
		
		
class POSNGramsFeature(Feature):
	def __init__(self, N=2):
		self._vectorizer = TfidfVectorizer(ngram_range=(1,2))
		self._initialized = False
		self._N = N;
	
	
	def extract_all(self, sentences):
		sentences = self.preprocess_all(sentences)
		posSentences = []
		for sentence in sentences:
			tokens = self.tokenize(sentence)
			posTags = nltk.pos_tag(tokens);
			posSentence = ""
			for t in posTags:
			    posSentence += "%s " % t[1]
			posSentences.append(posSentence[:-1])
			
		if not self._initialized:
			matrix = self._vectorizer.fit_transform(posSentences)
			self._initialized = True
		else:
			matrix = self._vectorizer.transform(posSentences)
		#print matrix.todense()
		return matrix



class BagOfWords(Feature):
	def __init__(self):
		self._vectorizer = TfidfVectorizer(ngram_range=(1,2))
		self._initialized = False

	def extract_all(self, sentences):
		sentences = self.preprocess_all(sentences)
		if not self._initialized:
			matrix = self._vectorizer.fit_transform(sentences)
			self._initialized = True
		else:
			matrix = self._vectorizer.transform(sentences)
		#print matrix.todense()
		return matrix


class CapFeature(Feature):
	def __init__(self):
		pass

	def extract(self, line):
		num_caps = len(re.findall(r'[A-Z]{3}[A-Z]*', line))
		num_lower = len(re.findall(r'[a-z]', line))
		return [float(num_caps)]

class RegexFeature(Feature):
	def __init__(self, regex):
		self._regex = regex

	def extract(self, line):
		return [len(re.findall(self._regex, line))]

class WordFeature(Feature):

	def __init__(self, words, is_binary=False):
		self._keywords = words
		self._stemmed_keywords = []
		stemmer = PorterStemmer()
		for word in words:
			self._stemmed_keywords.append(stemmer.stem_word(word))
		self._is_binary = is_binary

	def extract(self, line):
		
		words = self.tokenize(line.lower())
		count = 0.0
		stemmer = PorterStemmer()

		#print words
		#print "\n\n"
		for word in words:
			word = stemmer.stem_word(word)
			if word in self._stemmed_keywords:
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
		words = self.tokenize(line.lower())
		if self._word in words:
			pos = words.index(self._word)
			return [(1.0 / (1 + abs(self._position - pos)))]
		return [0.0]



def get_precision_recall(sentences, labels, predictions):
	feat = Feature()
	tp = fp = tn = fn = 0
	for i in range(len(labels)):
		label = labels[i]
		prediction = predictions[i]
		if label == 1:
			if prediction == 1:
				tp += 1
			else:
				print "FN: " + feat.preprocess(sentences[i])
				fn += 1
		else:
			if prediction == 0:
				tn += 1
			else:
				print "FP: " + feat.preprocess(sentences[i])
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

def evaluate(sentences, predictions, labels, name):
	print "------Results for " + name + " model------------"
	precision, recall = get_precision_recall(sentences, labels, predictions)
	print "precision: " + str(precision)
	print "recall: " + str(recall)

	F1 = (2.0 * precision * recall) / (precision + recall)

	print "F1: " + str(F1)

def run_baseline():
	sentences, labels = get_dev_data()
	results = baseline(sentences)
	evaluate(sentences, results, labels, "baseline")

def get_feature_values(sentences, features):
	all_values = []
	for feature in features:
		values = feature.extract_all(sentences)
		all_values.append(values)

	matrix = scipy.sparse.hstack(all_values)
	#input_values = tuple(all_values)
	#matrix = numpy.column_stack(input_values)
	return matrix


def get_features():
	feats = []
	you = WordFeature(["you", "u", "you're", "you've", "you'd", "your", "yours"])
	me = WordFeature(["me", "mine", "my", "i"])
	
	badwords = WordFeature(get_bad_words(), True)
	word_pos = WordPosition("you", 0)
	word_pos_ngram = POSNGramsFeature(3)
	cap = CapFeature()
	you_are = RegexFeature(r'([Yy]?o?u a?re?|[Yy]ou\'re) ')
	go_beginning = RegexFeature(r'^go ')
	exclaim = RegexFeature(r'!!+')
	bag_words = BagOfWords()
	bag_words_char = BagOfWords(1,2,'char')

	feats.extend([you, me, go_beginning, bag_words, bag_words_char, exclaim, badwords])
	return feats


run_baseline()
features = get_features()
train_sentences, train_labels = get_train_data()
matrix = get_feature_values(train_sentences, features)

print "-------------Got all train features - start training-----------------"

#print matrix[0]
#print train_labels
clf = svm.SVC(kernel='linear')
clf.fit(matrix, train_labels) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='linear', max_iter=-1, probability=False, shrinking=True,
tol=0.001, verbose=False)

print "------------------------Finished training-----------------"

dev_sentences, dev_labels = get_dev_data()

matrix_test = get_feature_values(dev_sentences, features)

print "-------------Got all test features - make predictions-----------------"

results = clf.predict(matrix_test)
#print results
evaluate(dev_sentences, results, dev_labels, "SVM")
