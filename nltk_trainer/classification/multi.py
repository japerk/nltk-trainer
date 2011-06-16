import collections, copy, itertools
from nltk.classify import ClassifierI, MultiClassifierI
from nltk.probability import DictionaryProbDist, MutableProbDist

class HierarchicalClassifier(ClassifierI):
	def __init__(self, root, label_classifiers):
		self.root = root
		self.label_classifiers = label_classifiers
		self._labels = copy.copy(self.root.labels())
		
		for label, classifier in self.label_classifiers.items():
			# label will never be returned from self.classify()
			self._labels.remove(label)
			self._labels.extend(classifier.labels())
	
	def labels(self):
		return self._labels
	
	def classify(self, feat):
		label = self.root.classify(feat)
		
		if label in self.label_classifiers:
			return self.label_classifiers[label].classify(feat)
		else:
			return label
	
	def prob_classify(self, feat):
		probs = self.root.prob_classify(feat)
		# passing in self.labels() ensures it doesn't have any of label_classifiers.keys()
		mult = MutableProbDist(probs, self.labels(), store_logs=False)
		
		for classifier in self.label_classifiers.values():
			pd = classifier.prob_classify(feat)
			
			for sample in pd.samples():
				mult.update(sample, pd.prob(sample), log=False)
		
		return mult

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