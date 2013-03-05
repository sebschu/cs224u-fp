from sentimenttokenizer import Tokenizer
from collections import defaultdict
import math
import csv
soadict = {} 
def get_bad_words():
	badwords = set(word.strip().lower() for word in (open("../data/badwords.txt").readlines()))
	return badwords

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

def get_pmi_for_word_pair(pair,pmidict): 
	if pair in pmidict: 
		return pmidict[pair]
	(w1,w2) = pair
	if (w2,w1) in pmidict: 
		return pmidict[(w2,w1)]
	return 0.0

def getSOAScores(sentences,pwords,nwords,context_size=10):
    """count_vect = CountVectorizer()
    sentencecounts = count_vect.fit_transform(sentences)
    print sentencecounts.get_features()
    """
    
    #sentences,labels = get_train_data()
   
    (unigram_prob , word_pair_prob) = BuildWordWordDict(sentences,context_size)
    pmiScores = defaultdict(float) 


    for (pair,prob_pair) in word_pair_prob.items(): 
    	w1,w2 = pair
    	pmiScores[(w1,w2)] = math.log(prob_pair/ (unigram_prob[w1] * unigram_prob[w2]),2)
 
    soa_score_dict = defaultdict(int) 
    for word in unigram_prob: 
    	score  = 0.0
    	for pword in pwords: 
    		sc = get_pmi_for_word_pair((pword,word),pmiScores)
    		score += sc
    		#if sc != 0: print pword,":",word,str(sc)
        for nword in nwords: 
            sc = get_pmi_for_word_pair((nword,word),pmiScores)
            score -= sc
           # if sc != 0.0: print nword,":",word,str(sc) 
        soa_score_dict[word] += score 
    #for (k,v) in soa_score_dict.items(): 
    # 	print k , "::" , v
    return soa_score_dict



def BuildWordWordDict(sentences,context_size):
	"Build probability dictionary for unigrams and word-pairs based on co-occurrence in context window"
	word_pairs_prob = defaultdict(float)
	unigrams_prob = defaultdict(float)
	numReviews = len(sentences)
	tokenizer = Tokenizer() 
	for s in sentences : 
		words = tokenizer.tokenize(s.lower().strip()) 
		reverse = list(words)
		reverse.reverse() 
		pairs_found = []
 		for i in range(len(words)) : 
			w1 = words[i]
			wr1 = reverse[i]
			for j in range(i+1,len(words)): 
				w2 = words[j]
				wr2 = reverse[j]
				if abs(i-j) > context_size: 
					 continue
			    #to capture right context pairs
				if w1 < w2 :
					pairs_found.append((w1,w2))
				else:
					pairs_found.append((w2,w1))
				#to capture left context pairs
				if wr1 < wr2 :
					pairs_found.append((wr1,wr2))
				else:
					pairs_found.append((wr2,wr1))
		for pair in set(pairs_found): 
			word_pairs_prob[pair] += 1
		for unigram in set(words):	
			unigrams_prob[unigram] += 1

	unigrams_prob = dict(map(lambda (k,v): (k,(v * 1.0)/numReviews), unigrams_prob.items()))
	word_pairs_prob = dict(map(lambda (k,v): (k,(v * 1.0)/numReviews), word_pairs_prob.items()))
	 
	#for (k,v) in word_pairs_prob.items():
	#	print k , "::" , v
	#for (k,v) in unigrams_prob.items():
	#	print k,":" , v
	return (unigrams_prob,word_pairs_prob)
def writeSOAtoDisk(soadict): 
	out = open("soadict","w")
	for (word,score) in soadict.items(): 
		print word 
		out.write(word) 
		out.write(",")
		print score 
		out.write(score.encode('utf-8'))
		out.write("\n")
	out.close()

def readSOAFromDisk(): 
	infile = open("soadict","r")
	soadict = defaultdict(float)
	for line in infile.readlines(): 
		print line
		word,score = line.split(",")
		soadict[word] = score 
	return soadict
def scoreSentence(tokens):
	scored =[]

	for token in tokens: 
		if token in soadict: 
			scored.append((token,soadict[token])) 
		else: 
			scored.append((token,"NA")) 
	return scored
if __name__ == "__main__":
	global soadict
 	pwords = ["good","nice", "excellent", "positive", "fortunate", "correct", "superior","smart"]
 	nwords = ["bad","nasty","poor","negative","unfortunate", "wrong","inferior","stupid"]
 #	pwords = get_opinion_words(positive="True")
 #	nwords = get_opinion_words(positive="False")
 	#print pwords 
 	#print nwords
 	#nwords = nwords.union(set(get_bad_words()))
 	sentences, labels = get_train_data()
 	soadict = getSOAScores(sentences,pwords,nwords,context_size=200) 
	#writeSOAtoDisk(soadict) 
	#soadict2 = readSOAFromDisk() 
	#print "word: good " , str(soadict2["good"])
	print scoreSentence(["this", "is", "a", "good", "story"])
