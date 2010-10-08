import argparse
from nltk.classify import MaxentClassifier
from nltk.metrics import BigramAssocMeasures

parser = argparse.ArgumentParser(description='Train a NLTK Classifier')

parser.add_argument('corpus',
	help='corpus name/path relative to nltk_data directory')
parser.add_argument('--filename', help='''filename/path for where to store the
	pickled classifier. the default is {corpus}_{algorithm}.pickle''')
parser.add_argument('--algorithm', default='naivebayes',
	choices=['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS,
	help='training algorithm to use. maxent used the default maxent training algorithm, either CG or iis')
parser.add_argument('--trace', default=0, type=int,
	help='how much trace output you want')

corpus_group = parser.add_argument_group('Training Corpus')
corpus_group.add_argument('--reader', choices=('plaintext', 'tagged'),
	default='plaintext',
	help='specify categorized plaintext or part-of-speech tagged corpus')
corpus_group.add_argument('--cat_pattern', default='(.+)/.+',
	help='regular expression pattern to identify categories based on file paths')
corpus_group.add_argument('--cat_file',
	help='relative path to a file containing category listings')
corpus_group.add_argument('--delimiter', default=' ',
	help='category delimiter for category file')
corpus_group.add_argument('--instances', default='files',
	choices=('sents', 'paras', 'files'),
	help='which groups of words represent a single training instance')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''what fraction of the corpus to use for training. the rest will be used for evaulation.
			the default is to use all of it, and to test the classifier against the training data.
			any other number < 1 will test against the remaining portion.''')

classifier_group = parser.add_argument_group('Classifier Type',
	'''A binary classifier has only 2 labels, and is the default classifier.
	A multi-class classifier chooses one of many possible labels.
	A multi-binary classifier choose zero or more labels by combining multiple
	binary classifiers, 1 for each label.''')
classifier_group.add_argument('--binary', action='store_true', default=False,
	help='train a binary classifier, or a multi-binary classifier if --multi is also given')
classifier_group.add_argument('--multi', action='store_true', default=False,
	help='train a multi-class classifier, or a multi-binary classifier if --binary is also given')

score_group = parser.add_argument_group('Feature Scoring')
score_group.add_argument('--score_fn', default='chi_sq',
	choices=[f for f in dir(BigramAssocMeasures) if not f.startswith('_')],
	help='scoring function for information gain and bigram collocations')
score_group.add_argument('--min_score', default=0, type=int,
	help='minimum score for a word to be included. if 0, all words are used.')
score_group.add_argument('--max-feats', default=0, type=int,
	help='maximum number of words to include, ordered by highest score. if 0, all words are used.')

maxent_group = parser.add_argument_group('Maxent Classifier',
	'These options only apply when a Maxent algorithm is chosen.')
maxent_group.add_argument('--max_iter', default=10, type=int,
	help='maximum number of training iterations')
maxent_group.add_argument('--min_ll', default=0, type=float,
	help='stop classification when average log-likelihood is less than this')
maxent_group.add_argument('--min_lldelta', default=0.1, type=float,
	help='stop classification when the change in average log-likelihood is less than this')

decisiontree_group = parser.add_argument_group('Decision Tree Classifier',
	'These options only apply when the DecisionTree algorithm is chosen')
decisiontree_group.add_argument('--entropy_cutoff', default=0.05, type=float)
decisiontree_group.add_argument('--depth_cutoff', default=100, type=int)
decisiontree_group.add_argument('--support_cutoff', default=10, type=int)

args = parser.parse_args()