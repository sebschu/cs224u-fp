"""
  adapted from http://norvig.com/spell-correct.html
"""

import re, collections

class SpellChecker():

  def __init__(self):
    self._NWORDS = self.train(self.words(file('../data/badwords.txt').read()))
    self._alphabet = 'abcdefghijklmnopqrstuvwxyz'
    self._cache = collections.defaultdict()
  def words(self,text): return re.findall('[a-z]+', text.lower()) 

  def train(self, features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
    
  def edits1(self,word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in self._alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in self._alphabet]
   return set(deletes + transposes + replaces + inserts)

  def known_edits2(self,word):
    return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self._NWORDS)

  def known(self,words): return set(w for w in words if w in self._NWORDS)

  def correct(self,word):
    if not word in self._cache:
      candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
      self._cache[word] = max(candidates, key=self._NWORDS.get)
    return self._cache[word]