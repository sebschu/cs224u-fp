import scipy
from sklearn import svm
from sklearn.svm import SVC

import csv

reader = csv.reader(open("../data/kaggle/train.csv"), delimiter=',')
i = 0
for row in reader:
	print row
	i+=1
	if i > 10: break

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y) 
print "hello"
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, shrinking=True,
tol=0.001, verbose=False)


print clf.predict([[2.,2.], [0, -1]])
