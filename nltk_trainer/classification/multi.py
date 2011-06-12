import collections, itertools
from nltk.classify import ClassifierI, MultiClassifierI
from nltk.probability import DictionaryProbDist

class AvgProbClassifier(ClassifierI):
	def __init__(self, classifiers):
		self._classifiers = classifiers
		self._labels = sorted(set(itertools.chain(*[c.labels() for c in classifiers])))
	
	def labels(self):
		return self._labels
	
	def classify(self, feat):
		return self.prob_classify(feat).max()
	
	def prob_classify(self, feat):
		label_probs = collections.defaultdict(list)
		
		for classifier in self._classifiers:
			cprobs = classifier.prob_classify(feat)
			
			for label in cprobs.samples():
				label_probs[label].append(cprobs.prob(label))
		
		avg_probs = {}
		
		for label, probs in label_probs.items():
			avg_probs[label] = float(sum(probs)) / len(probs)
		
		return DictionaryProbDist(avg_probs)

class MultiBinaryClassifier(MultiClassifierI):
	def __init__(self, label_classifiers):
		self._label_classifiers = label_classifiers
		self._labels = sorted(label_classifiers.keys())
	
	def labels(self):
		return self._labels
	
	def classify(self, feats):
		lbls = set()
		
		for label, classifier in self._label_classifiers.iteritems():
			if classifier.classify(feats) is True:
				lbls.add(label)
		
		return lbls
	
	@classmethod
	def train(cls, labels, multi_label_feats, trainf, **train_kwargs):
		labelset = set(labels)
		label_feats = collections.defaultdict(list)
		
		for feat, multi_labels in multi_label_feats:
			for label in multi_labels:
				label_feats[label].append((feat, True))
			
			for label in labelset - set(multi_labels):
				label_feats[label].append((feat, False))
		
		label_classifiers = {}
		
		for label, feats in label_feats.iteritems():
			label_classifiers[label] = trainf(feats, **train_kwargs)
		
		return cls(label_classifiers)