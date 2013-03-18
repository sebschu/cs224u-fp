import scipy
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
import numpy
import nltk
#from nltk import word_tokenize
from nltk.metrics import edit_distance
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import defaultdict
import csv
import re
from sentimenttokenizer import Tokenizer
from spellchecking import SpellChecker
import sentimentorientation 
from optparse import OptionParser
RESULTS_FILE = "results.txt"

class Feature(object):
	tokens_cache = dict()
	#checker = SpellChecker()
	sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  
	def __init__(self):
		pass

	def preprocess(self, sentence):
		sentence = re.sub(r'"', "", sentence)
		sentence = re.sub(r'(\w)(\1\1\1+)', r'\1', sentence)
		sentence = re.sub(r'_', " ", sentence)
		sentence = re.sub(r'^@[a-z0-9_A-Z-]+', "NAMEMENTION", sentence)
		sentence = re.sub(r'@', "a", sentence)
		sentence = re.sub(r'\\+[ux][0-9a-f]+', " ", sentence)
		sentence = re.sub(r'\\+[nt]',' ',sentence)
		sentence = re.sub(r'\\+[\']','\'',sentence)
		sentence = re.sub(r'&\w+;',' ',sentence)
		sentence = re.sub(r'<([^>]+)>', ' ', sentence)
		sentence = re.sub(r'(https?:\/\/).*? ', ' ', sentence)
		sentence = re.sub(r' +', ' ', sentence)
		sentence = re.sub("yall", "you all", sentence)
		return sentence

	def preprocess_all(self, sentences):
		new_sentences = []
		for sentence in sentences:
			new_sentences.append(self.preprocess(sentence))
		return new_sentences

	def tokenize(self, sentence):
		sentence = self.preprocess(sentence)
		if sentence not in Feature.tokens_cache:
			tokenizer = Tokenizer()
			Feature.tokens_cache[sentence] = tokenizer.tokenize(sentence)
		return Feature.tokens_cache[sentence]

	def extract_all(self, sentences):
		all_values = []
		for sentence in sentences:
			values = self.extract(sentence)
			all_values.append(values)
		return scipy.sparse.coo_matrix(all_values)

class BagOfWords(Feature):
	
	def name(self):
		return "BagOfWords with mn=" + str(self._mn) + ", mx=" + str(self._mx) + ", analyzertype=" + self._analyzertype + ", numFeatures=" + str(self._numFeatures)
		
	def __init__(self,numFeatures, mn=1, mx=2, analyzertype='word'):
		self._tokenizer = Tokenizer()	
		if analyzertype == 'word':
			self._vectorizer = TfidfVectorizer(ngram_range=(mn,mx),analyzer=analyzertype)
		else:
			self._vectorizer = TfidfVectorizer(ngram_range=(mn,mx),analyzer=analyzertype)
		self._initialized = False
		self._mn = mn
		self._mx = mx
		self._analyzertype = analyzertype
		self._numFeatures = numFeatures
		self._ch2 = SelectKBest(chi2, k=numFeatures)
		self._ch22 = SelectKBest(chi2, k=10)

	def extract_all(self, sentences,train,labels):
		sentences = self.preprocess_all(sentences)
		if not self._initialized:
			matrix = self._vectorizer.fit_transform(sentences)
			self._initialized = True
		else:
			matrix = self._vectorizer.transform(sentences)
		#print matrix.todense()
		if train:
			self._ch22.fit(matrix,labels)
			indices = self._ch22.get_support(indices=True)
			featureNames = self._vectorizer.get_feature_names()
			for i in indices:
				print featureNames[i]
		if self._numFeatures < matrix.shape[1]:
			if train:
				matrix = self._ch2.fit_transform(matrix, labels)
			else:
				matrix = self._ch2.transform(matrix)
		return matrix


class CapFeature(Feature):
	
	def name(self):
		return "CapFeature"
	
	def __init__(self):
		pass

	def extract(self, line):
		count = 0
		words = line.split()
		for word in words:
			if word.isupper():
				count += 1
		return [count]

class RegexFeature(Feature):

	def name(self):
		return "RegexFeature, regex=" + str(self._regex)
	
	def __init__(self, regex):
		self._regex = regex

	def extract(self, line):
		return [len(re.findall(self._regex, line))]

class WordFeature(Feature):

	def name(self):
		return "WordFeature"

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
		for word in words:
			word = stemmer.stem_word(word)
			if word in self._stemmed_keywords:
				count += 1
		if self._is_binary:
			return [1] if count > 0 else [0]
		else:
			return [count]

class WordPairFeature(Feature):

	def name(self):
		return "WordPairFeature"

	def __init__(self, badwords,youwords):
		self._badwords = badwords
		self._stemmed_badwords = []
		self._youwords = youwords
		self._part_of_badword = {}    #cache of words that start or end with offensive content
		stemmer = PorterStemmer()
		for word in badwords:
			self._stemmed_badwords.append(stemmer.stem_word(word))

	def isWordPartOf(self,word,wordlist): 
		"""
		return True if word starts or ends with a word from wordlist
		"""
		for w in wordlist:
			if w in self._part_of_badword: 
				return True    	 
				if w.startswith(word) or w.endswith(word):
					self._part_of_badword[w] = True 
					return True
		return False

	def extract(self, line):
		"""
		find word pairs that co-occur and extract # of minimum distance word pairs in the line
		"""
		words = self.tokenize(line.lower())
		count = 0.0
		stemmer = PorterStemmer()
		bad_indices = [] 
		you_indices = [] 
		for i in range(len(words)):
			word = words[i] 
			if word in self._youwords: 
				you_indices.append(i)
			word = stemmer.stem_word(word)
			if word  in self._stemmed_badwords or self.isWordPartOf(word,self._badwords): 
				bad_indices.append(i)
			
		 
		if not bad_indices or not you_indices: 
			return [-1]
		else: 
			distances = [] 
			for bindex in bad_indices:
				for yindex in you_indices: 
					distances.append(abs(bindex - yindex))
			mn = min(distances)
			count = sum([1 for d  in distances if d == mn])
			return [1]


def get_precision_recall(sentences, labels, probs, threshold, options):
	feat = Feature()
	you_are = ["you are", "u r ", "ur ", "your ", "you're", "go ", "get "]
	badwords = get_bad_words()
	tp = fp = tn = fn = 0
	
	for i in range(len(labels)):
		sentence = feat.preprocess(sentences[i]).lower()
		forcePositive = False
		if options.useRules:
			if probs[i][1] < threshold and probs[i][1] > 0.3:
				for y in you_are:
					if (sentence.startswith(y)):
						forcePositive = True
				if not forcePositive and probs[i][1] > 0.4:
					for w in feat.tokenize(sentence):
						if w in badwords:
							forcePositive = True
							break
		label = labels[i]
		if label == 1:
			if forcePositive or probs[i][1] >= threshold:
				tp += 1
			else:
				#if probs[i][1] > 0.3:
				if options.verbose:
					print "FN (" + str(probs[i][1]) + "): " + sentence
				fn += 1
		else:
			if not forcePositive and probs[i][1] < threshold:
				tn += 1
			else:
				if options.verbose:
					print "FP (" + str(probs[i][0]) + "): " + sentence
				fp += 1
	if (tp + fp) > 0:
		precision = float(tp) / (tp + fp)
	else:
		precision = 1
	if (tp + fn) > 0:
		recall = float(tp) / (tp + fn)
	else:
		recall = 0
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
	
def get_other_data():
	return get_data("../data/other_insults.csv")

def get_yt_data():
	return get_data("../data/other/yt.csv")

def get_test_data():
	return get_data("../data/kaggle/test.csv")
	
def get_sentences(comments):
	sentence_index = []
	sentence_count = []
	sentences = []
	for comment in comments:
		sents = Feature.sentence_tokenizer.tokenize(comment)
		sentence_index.append(len(sentences))
		count = 0
		for s in sents:
			s = s.strip()
			if len(s) > 2:
				sentences.append(s)
				count = count + 1
		if count < 1:
			count = 1
			sentences.append(comment)
		sentence_count.append(count)
	return [sentences, sentence_index, sentence_count]
	
	
def get_opinion_words(positive="True",num=2000):
	words = []
	if positive: 
		words = open("../data/positive-words.txt").readlines()
	else:
		words = open("../data/negative-words.txt").readlines()

	opinion_words = [] 
	for w in words : 
		if w.startswith(";") or len(w) == 0: 
		   continue
		opinion_words.append(w.lower().strip())
	return set(opinion_words)

def baseline(sentences):
	predictions = []
	badwords = get_bad_words()
	for line in sentences:
		words = set(line.lower().split())
		if not words & badwords:
			predictions.append([1,0])
		else:
			predictions.append([0,1])
	return predictions

def evaluate(sentences, labels, name, features, probs, options):
	if name == "baseline": return
	print "------------------------Results------------------------"
	print "Model:" + name
	print "Rules: " + str(options.useRules)
	print "Sentences: " + str(options.useSentences)
	print "Data: " + options.data 
	print "-------------------------------------------------------"
	print "Threshold\tPrecision\tRecall\tF1"
	if options.precisionRecallCurve:
		lower_limit = -0.05
		upper_limit = 0.95
	else:
		lower_limit = 0.45
		upper_limit = 0.5		
	i = upper_limit
	while i > lower_limit:
		precision, recall = get_precision_recall(sentences, labels, probs, i, options)
		F1 = (2.0 * precision * recall) / (precision + recall)
		print str(max(0,i)) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(F1)
		i = i - 0.05

def run_baseline(options):
	sentences, labels = get_yt_data()
	results = baseline(sentences)
	evaluate(sentences, labels, "baseline", [], results, options)

def get_feature_values(sentences, features, train=True, labels=None, numFeatures=10000):
	all_values = []
	i = len(features)
	for feature in features:
		print str(i) + " features left to get"
		i -= 1
		if isinstance(feature,BagOfWords) :
			values  = feature.extract_all(sentences,train,labels)
		else:
		    values = feature.extract_all(sentences)
		all_values.append(values)

	matrix = scipy.sparse.hstack(all_values)
	return matrix


def get_features():
	feats = []
	youlist = ["ur","you", "u", "you're","you've","you'd","your","yours","yourself"]
	me = WordFeature(["me","my","i","mine"])
	cap = CapFeature()
	go_beginning = RegexFeature(r'^\s*[Gg]o ')
	get_beginning = RegexFeature(r'^\s*[Gg]et ')
	bag_words2 = BagOfWords(10000,1,2,'char')
	bag_words = BagOfWords(10000,1,2)
	#bag_words3 = BagOfWords(6000,2,2)
	#bag_words4 = BagOfWords(100,3,3)
	#bag_words5 = BagOfWords(1000,4,4,'char')
	#bag_words6 = BagOfWords(1000,5,5,'char')
	#bag_words7 = BagOfWords(2000,3,3,'char')

	word_pair = WordPairFeature(get_bad_words(),youlist)
	feats.extend([word_pair,bag_words,bag_words2,cap])
	return feats


def parse_options():
	parser = OptionParser()
	parser.add_option("-l", "--learningCurve", default=False, action="store_true", dest="learningCurve", help="generate data for learning curve")
	parser.add_option("-s", "--sentences", dest="useSentences", help="use sentences for testing instead of complete comment", default=False, action="store_true")
	parser.add_option("-r", "--rules", dest="useRules", help="use hand-written rules", default=False, action="store_true")  
	parser.add_option("-p", "--precisionRecallCurve", dest="precisionRecallCurve", help="generate data for precision/recall curve", default=False, action="store_true")  
	parser.add_option("-v", "--verbose", dest="verbose", help="print FP/FN", default=False, action="store_true")  
	parser.add_option("-d", "--data", dest="data", help="test data to use [dev,test,yt]", default="dev")  
	(options, args) = parser.parse_args()
	return options


def run_subset(trainComments, trainLabels, sampleSize, options):
	 
	print "Sample Size: " + str(sampleSize)
	
	train_comments = trainComments[0:sampleSize-1]
	train_labels = trainLabels[0:sampleSize-1]
		
	features = get_features()
	print "-------------Start getting train features----------------------------"
	matrix = get_feature_values(train_comments, features, True, train_labels)
	
	print "-------------Got all train features - Starting Training-----------------"
	clf1 = svm.SVC(kernel='linear', probability=True,C=1.0)
	clf1.fit(matrix, train_labels)
	#maxent = LogisticRegression(penalty='l2', C=3.0, class_weight={0:1,1:4})
	#maxent.fit(matrix, train_labels)
	
	print "------------------------Finished training-----------------"

	if options.data == 'dev':
		dev_comments, dev_labels = get_dev_data()
	elif options.data == 'test':
		dev_comments, dev_labels = get_test_data()
	elif options.data == 'yt':
		dev_comments, dev_labels = get_yt_data()
	dev_sentences, dev_sentence_index, dev_sentence_counts = get_sentences(dev_comments)
	if options.useSentences:
		matrix_test = get_feature_values(dev_sentences, features, False)
	else:
		matrix_test = get_feature_values(dev_comments, features, False)
		
	print "-------------Got all test features - make predictions-----------------"
	
	sentence_result_probs = clf1.predict_proba(matrix_test)
	# * 0.9 + 0.1 * maxent.predict_proba(matrix_test)
	
	if options.useSentences:
		result_probs = []	
		for i in range(0,len(dev_sentence_index)):
			result_probs.append([1.0,0.0])
			totalProb = 0.0
			for j in range(dev_sentence_index[i],  dev_sentence_index[i] + dev_sentence_counts[i]):
				totalProb = totalProb + sentence_result_probs[j][1]
				if result_probs[i][1] < sentence_result_probs[j][1]:
					result_probs[i] = sentence_result_probs[j]
			totalProb = totalProb / float(dev_sentence_counts[i])
			if totalProb < 0.2 and result_probs[i][1] < 0.9:
				result_probs[i] = [1-totalProb,totalProb]
	else:
		result_probs = sentence_result_probs
	
	#kf = cross_validation.Bootstrap(len(dev_comments), train_size=0.8, n_iter=10)
	#for train_index, test_index in kf:
	#	dev_comments_sample = numpy.array(dev_comments)[train_index]
	#	dev_labels_sample = numpy.array(dev_labels)[train_index]
	# 	result_probs_sample = numpy.array(result_probs)[train_index]
	evaluate(dev_comments, dev_labels, "SVM", features, result_probs, options)




if __name__ == "__main__":
	train_comments, train_labels = get_train_data()
	options = parse_options()
	run_baseline(options)
	if options.learningCurve:
		for i in range(500,len(train_labels), 500):
			run_subset(train_comments, train_labels, i, options)
	run_subset(train_comments, train_labels, len(train_labels), options)
	
	
	
	
