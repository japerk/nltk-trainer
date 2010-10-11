import collections
from nltk.classify import MultiClassifierI

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