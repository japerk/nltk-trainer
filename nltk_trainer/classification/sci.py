import scipy.sparse
from scikits.learn.base import BaseEstimator
from scikits.learn.feature_extraction.text.dense import BaseCountVectorizer
from scikits.learn.svm.sparse import LinearSVC
from scikits.learn.pipeline import Pipeline
from nltk.classify import ClassifierI

class BagOfWordsAnalyzer(BaseEstimator):
	def analyze(self, feats):
		# this will work for feat dicts and lists of tokens
		return feats

BOWAnalyzer = BagOfWordsAnalyzer()

class BagOfWordsVectorizer(BaseCountVectorizer):
	def __init__(self, analyzer=BOWAnalyzer, max_df=None):
		BaseCountVectorizer.__init__(self, analyzer=analyzer, max_df=max_df)
	
	def _term_count_dicts_to_matrix(self, term_count_dicts, vocabulary):
		i_indices, j_indices, values = [], [], []
		
		for i, term_count_dict in enumerate(term_count_dicts):
			for term in term_count_dict.iterkeys(): # ignore counts
				j = vocabulary.get(term)
				
				if j is not None:
					i_indices.append(i)
					j_indices.append(j)
					values.append(1)
			
			term_count_dict.clear()
		
		shape = (len(term_count_dicts), max(vocabulary.itervalues()) + 1)
		return scipy.sparse.coo_matrix((values, (i_indices, j_indices)),
			shape=shape, dtype=self.dtype)

class ScikitsClassifier(ClassifierI):
	def __init__(self, pipeline, target_names):
		self.pipeline = pipeline
		self.target_names = target_names
	
	def labels(self):
		return self.target_names
	
	def classify(self, featureset):
		return self.target_names[self.pipeline.predict([featureset])[0]]
	
	@classmethod
	def train(cls, labeled_featuresets):
		train, target_labels = zip(*labeled_featuresets)
		target_names = sorted(set(target_labels))
		targets = [target_names.index(l) for l in target_labels]
		
		pipeline = Pipeline([
			('bow', BagOfWordsVectorizer()),
			('clf', LinearSVC(C=1000)),
		])
		
		pipeline.fit(train, targets)
		return cls(pipeline, target_names)