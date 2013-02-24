import scipy
from sklearn import svm
from sklearn.svm import SVC
import numpy
import nltk
#from nltk import word_tokenize
from nltk.metrics import edit_distance
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import defaultdict
import csv
import re

RESULTS_FILE = "results.txt"

class Feature(object):
	tokens_cache = dict()

	def __init__(self):
		pass

	def preprocess(self, sentence):
		sentence = re.sub(r'"', "", sentence)
		sentence = re.sub(r'(\w)(\1\1\1+)', r'\1', sentence)
		sentence = re.sub(r'_', " ", sentence)
		#sentence = re.sub(r'-', " ", sentence)
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
		sentence = self.preprocess(sentence)
		if sentence not in Feature.tokens_cache:
			tokenizer = TreebankWordTokenizer()
			Feature.tokens_cache[sentence] = tokenizer.tokenize(sentence)
		return Feature.tokens_cache[sentence]

	def extract_all(self, sentences):
		all_values = []
		for sentence in sentences:
			values = self.extract(sentence)
			all_values.append(values)
		return scipy.sparse.coo_matrix(all_values)
		
		
class POSNGramsFeature(Feature):
	
	def name(self):
		return "POSNGrams with N=" + str(self._N)
	
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
			#print posSentences
			
		if not self._initialized:
			matrix = self._vectorizer.fit_transform(posSentences)
			self._initialized = True
		else:
			matrix = self._vectorizer.transform(posSentences)
		#print matrix.todense()
		return matrix

class WordTagBigram(Feature):
	
	def name(self):
		return "WordTagBigram with N=" + str(self._N) + " and word=" + self._word
	
	def __init__(self, word, N=4):
		self._vectorizer = TfidfVectorizer(ngram_range=(1,2))
		self._initialized = False
		self._word = word
		self._N = N

	def extract_all(self, sentences):
		sentences = self.preprocess_all(sentences)
		all_phrases = []
		for sentence in sentences:
			phrase = ""
			tokens = self.tokenize(sentence)
			if self._word in tokens:
				
				index = tokens.index(self._word)
				posTags = nltk.pos_tag(tokens)
				word_tag = posTags[index]
				if index < len(tokens) - self._N:
					for i in range(1, self._N+1):
						phrase += posTags[index+i][1] + " "
					#print phrase
					#print posTags[index+1][0] + " " + posTags[index+2][0]
					
			all_phrases.append(phrase)
		if not self._initialized:
			matrix = self._vectorizer.fit_transform(all_phrases)
			self._initialized = True
		else:
			matrix = self._vectorizer.transform(all_phrases)
		#print matrix.todense()
		return matrix




class BagOfWords(Feature):
	
	def name(self):
		return "BagOfWords with mn=" + str(self._mn) + ", mx=" + str(self._mx) + ", analyzertype=" + self._analyzertype + ", numFeatures=" + str(self._numFeatures)
		
	def __init__(self,numFeatures, mn=1, mx=2, analyzertype='word'):
		self._vectorizer = TfidfVectorizer(ngram_range=(mn,mx),analyzer=analyzertype)
		self._initialized = False
		self._mn = mn
		self._mx = mx
		self._analyzertype = analyzertype
		self._numFeatures = numFeatures
		self._ch2 = SelectKBest(chi2, k=numFeatures)

	def extract_all(self, sentences,train,labels):
		sentences = self.preprocess_all(sentences)
		if not self._initialized:
			matrix = self._vectorizer.fit_transform(sentences)
			self._initialized = True
		else:
			matrix = self._vectorizer.transform(sentences)
		#print matrix.todense()
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
		#num_caps = len(re.findall(r'[A-Z]{3}[A-Z]*', line))
		#num_lower = len(re.findall(r'[a-z]', line))
		#return [float(num_caps)]

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
			#print line 
			#print bad_indices
			#print you_indices
			distances = [] 
			for bindex in bad_indices:
				for yindex in you_indices: 
					distances.append(abs(bindex - yindex))
			#print distances
			mn = min(distances)
			count = sum([1 for d  in distances if d == mn])
			#return [(count *1.0)* mn/len(line)]		
			return [1]

class WordPosition(Feature):
	
	def name(self):
		return "WordPosition"
	
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

def evaluate(sentences, predictions, labels, name, features):

	if name == "baseline": return
	print "------Results for " + name + " model------------"
	precision, recall = get_precision_recall(sentences, labels, predictions)
	print "precision: " + str(precision)
	print "recall: " + str(recall)

	F1 = (2.0 * precision * recall) / (precision + recall)

	print RESULTS_FILE
	fd = open(RESULTS_FILE, 'r')
	all_lines = fd.readlines()
	old_best = all_lines[0]
	print old_best
	old_best_int = float(old_best)
	if F1 > old_best_int:
		fd = open(RESULTS_FILE, 'w')
		print "Got a new best F1 of " + str(F1)
		fd.write(str(F1) + "\n")
		for feat in features:
			fd.write("* " + feat.name() + "\n")
		fd.write("\n####################################################################\n\n")
		for l in all_lines:
			fd.write(l)
	print "F1: " + str(F1)

def run_baseline():
	sentences, labels = get_dev_data()
	results = baseline(sentences)
	evaluate(sentences, results, labels, "baseline", [])

def get_feature_values(sentences, features, train=True, labels=None, numFeatures=10000):
	all_values = []
	i = len(features)
	for feature in features:
		print str(i) + " features left to get"
		i -= 1
		if isinstance(feature,BagOfWords):
			values  = feature.extract_all(sentences,train,labels)
		else:
		    values = feature.extract_all(sentences)
		all_values.append(values)

	matrix = scipy.sparse.hstack(all_values)
	#input_values = tuple(all_values)
	#matrix = numpy.column_stack(input_values)
	return matrix


def get_features():
	feats = []
	youlist = ["ur","you", "u", "you're","you've","you'd","your","yours","yourself"]
	you = WordFeature(youlist)
	me = WordFeature(["me","my","i","mine"])

	insults = WordFeature(["moron","iq","idiot","dumb","stupid","fool","dimwit","specimen"])
	badwords = WordFeature(get_bad_words(), True)
	word_pos = WordPosition("you", 0)
	word_pos_ngram = POSNGramsFeature(3)
	cap = CapFeature()
	you_are = RegexFeature(r'([Yy]?o?u a?re?|[Yy]ou\'re) ')
	go_beginning = RegexFeature(r'^go ')
	exclaim = RegexFeature(r'!!+')
	question = RegexFeature(r'\?')
	bag_words = BagOfWords(10000)
	bag_words2 = BagOfWords(10000,1,2,'char')

	word_tag = WordTagBigram("you")
	word_pair = WordPairFeature(get_bad_words(),youlist)

	feats.extend([word_pair,bag_words,bag_words2,me,cap,go_beginning])
	return feats


run_baseline()
features = get_features()
train_sentences, train_labels = get_train_data()
print "-------------Start getting train features----------------------------"
matrix = get_feature_values(train_sentences, features, True, train_labels)

print "-------------Got all train features - start training-----------------"

#print matrix[0]
#print train_labels
clf = svm.SVC(kernel='linear')
clf.fit(matrix, train_labels) 


clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='linear', max_iter=-1, probability=True, shrinking=True,
tol=0.001, verbose=False)

print "------------------------Finished training-----------------"

dev_sentences, dev_labels = get_dev_data()

matrix_test = get_feature_values(dev_sentences, features, False)

print "-------------Got all test features - make predictions-----------------"

results = clf.predict(matrix_test)
#print results
evaluate(dev_sentences, results, dev_labels, "SVM", features)
