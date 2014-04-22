import collections, copy, itertools
from nltk.classify import ClassifierI, MultiClassifierI
from nltk.probability import DictionaryProbDist, MutableProbDist
from nltk_trainer import iteritems

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
		'''Return the label with the most agreement among classifiers'''
		label_freqs = collections.Counter()
		
		for classifier in self._classifiers:
			label_freqs[classifier.classify(feat)] += 1
		
		return label_freqs.most_common(1)[0][0]
	
	def prob_classify(self, feat):
		'''Return ProbDistI of averaged label probabilities.'''
		label_probs = collections.defaultdict(list)
		
		for classifier in self._classifiers:
			try:
				cprobs = classifier.prob_classify(feat)
				
				for label in cprobs.samples():
					label_probs[label].append(cprobs.prob(label))
			except NotImplementedError:
				# if we can't do prob_classify (like for DecisionTree)
				# assume 100% probability from classify
				label_probs[classifier.classify(feat)].append(1)
		
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
		
		for label, classifier in iteritems(self._label_classifiers):
			if classifier.classify(feats) is True:
				lbls.add(label)
		
		return lbls
	
	@classmethod
	def train(cls, labels, multi_label_feats, trainf, **train_kwargs):
		labelset = set(labels)
		label_feats = collections.defaultdict(list)
		pos_label_feats = collections.defaultdict(set)
		
		for feat, multi_labels in multi_label_feats:
			for label in multi_labels:
				label_feats[label].append((feat, True))
				# dicts are unhashable, so use a normalized tuple of key-values
				pos_label_feats[label].add(tuple(sorted(feat.items())))
			
			for label in labelset - set(multi_labels):
				label_feats[label].append((feat, False))
		
		for label in label_feats.keys():
			feats = []
			# this re-creates the feats list by ignoring any negative feat dicts
			# that are also in pos_label_feats[label] so we don't create
			# training conflicts
			for feat, l in label_feats[label]:
				if l or tuple(sorted(feat.items())) not in pos_label_feats[label]:
					feats.append((feat, l))
			
			label_feats[label] = feats
		
		label_classifiers = {}
		
		for label, feats in iteritems(label_feats):
			label_classifiers[label] = trainf(feats, **train_kwargs)
		
		return cls(label_classifiers)