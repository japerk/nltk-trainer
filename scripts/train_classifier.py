import argparse, collections, itertools, math, re
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import stopwords
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, CategorizedTaggedCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from nltk.metrics import BigramAssocMeasures, f_measure, masi_distance, precision, recall
from nltk.util import bigrams

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Train a NLTK Classifier')

parser.add_argument('corpus',
	help='corpus name/path relative to nltk_data directory')
parser.add_argument('--filename', help='''filename/path for where to store the
	pickled classifier. the default is {corpus}_{algorithm}.pickle''')
parser.add_argument('--algorithm', default='NaiveBayes',
	choices=['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS,
	help='training algorithm to use. maxent used the default maxent training algorithm, either CG or iis')
parser.add_argument('--trace', default=1, type=int,
	help='how much trace output you want. defaults to 1, 0 is no trace output.')

corpus_group = parser.add_argument_group('Training Corpus')
corpus_group.add_argument('--reader', choices=('plaintext', 'tagged'),
	default='plaintext',
	help='specify categorized plaintext or part-of-speech tagged corpus')
corpus_group.add_argument('--cat_pattern', default='(.+)/.+',
	help='''regular expression pattern to identify categories based on file paths.
	if cat_file is also given, this pattern is used to identify corpus file ids.''')
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

feat_group = parser.add_argument_group('Feature Extraction',
	'The default is to lowercase every word and filter out stopwords')
feat_group.add_argument('--bigrams', action='store_true', default=False,
	help='include bigrams as features')
feat_group.add_argument('--no-lowercase', action='store_true', default=False,
	help="don't lowercase every word")
feat_group.add_argument('--filter-stopwords', default='english',
	choices=['no']+stopwords.fileids(),
	help='stopwords to filter, or "no" if want to keep stopwords')

score_group = parser.add_argument_group('Feature Scoring',
	'The default is no scoring, all words are included as features')
score_group.add_argument('--score_fn', default='chi_sq',
	choices=[f for f in dir(BigramAssocMeasures) if not f.startswith('_')],
	help='scoring function for information gain and bigram collocations')
score_group.add_argument('--min_score', default=0, type=int,
	help='minimum score for a word to be included. if 0, all words are used.')
score_group.add_argument('--max-feats', default=0, type=int,
	help='maximum number of words to include, ordered by highest score. if 0, all words are used.')

eval_group = parser.add_argument_group('Classifier Evaluation',
	'''The default is to test the classifier against the unused fraction of the
	corpus, or against the entire corpus if the whole corpus is used for training.''')
eval_group.add_argument('--no-eval', action='store_true', default=False,
	help="don't do any evaluation")
eval_group.add_argument('--no-accuracy', action='store_true', default=False,
	help="don't evaluate accuracy")
eval_group.add_argument('--no-precision', action='store_true', default=False,
	help="don't evaluate precision")
eval_group.add_argument('--no-recall', action='store_true', default=False,
	help="don't evaluate recall")
eval_group.add_argument('--no-fmeasure', action='store_true', default=False,
	help="don't evaluate f-measure")
eval_group.add_argument('--no-masi-distance', action='store_true', default=False,
	help="don't evaluate masi distance (only applies to a multi binary classifier)")

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

###################
## corpus reader ##
###################

reader_class = {
	'plaintext': CategorizedPlaintextCorpusReader,
	'tagged': CategorizedTaggedCorpusReader
}

reader_args = []
reader_kwargs = {}

if args.cat_file:
	reader_kwargs['cat_file'] = args.cat_file
	
	if args.delimiter and args.delimiter != ' ':
		reader_kwargs['delimiter'] = args.delimiter
	
	if args.cat_pattern:
		reader_args.append(args.cat_pattern)
	else:
		reader_args.append('.+/.+')
elif args.cat_pattern:
	reader_args.append(args.cat_pattern)
	reader_kwargs['cat_pattern'] = re.compile(args.cat_pattern)

categorized_corpus = LazyCorpusLoader(args.corpus, reader_class[args.reader],
	*reader_args, **reader_kwargs)
labels = categorized_corpus.categories()
nlabels = len(labels)

if args.trace:
	print '%d labels: %s' % (nlabels, labels)

if not nlabels:
	raise ValueError('corpus does not have any categories')
elif nlabels == 1:
	raise ValueError('corpus must have more than 1 category')
elif nlabels == 2 and args.multi:
	raise ValueError('corpus must have more than 2 categories if --multi is specified')

#####################
## text extraction ##
#####################

if args.filter_stopwords == 'no':
	stopset = set()
else:
	stopset = set(stopwords.words(args.filter_stopwords))

def norm_sent(words):
	if not args.no_lowercase:
		words = [w.lower() for w in words]
	
	if stopset:
		words = [w for w in words if w.lower() not in stopset]
	
	if args.bigrams:
		return words + bigrams(words)
	else:
		return words

def label_sents(label):
	for sent in categorized_corpus.sents(categories=[label]):
		yield norm_sent(sent)

def label_paras(label):
	for para in categorized_corpus.paras(categories=[label]):
		yield list(itertools.chain(*[norm_sent(sent) for sent in para]))

def label_files(label):
	for fileid in categorized_corpus.fileids(categories=[label]):
		sents = categorized_corpus.sents(fileids=[fileid])
		yield list(itertools.chain(*[norm_sent(sent) for sent in sents]))

label_instance_function = {
	'sents': label_sents,
	'paras': label_paras,
	'files': label_files
}

lif = label_instance_function[args.instances]
label_instances = {}

for label in labels:
	label_instances[label] = list(lif(label))
	
	if args.trace:
		print '%s has %d instances' % (label, len(label_instances[label]))

##################
## word scoring ##
##################

def bag_of_words(words):
	return dict([(word, True) for word in words])

def bag_of_words_in_set(words, wordset):
	return bag_of_words(set(words) & wordset)

if args.min_score or args.max_feats:
	# TODO: score words by label and create top words set to use with
	# bag_of_words_in_set
	pass

featx = bag_of_words

##############################
## training & testing feats ##
##############################

# TODO: this will have to be modified for --multi --binary classifier

train_feats = []
test_feats = []

for label, instances in label_instances.iteritems():
	l = len(instances)
	labeled_instances = [(featx(i), label) for i in instances]
	
	if args.fraction != 1.0:
		cutoff = int(math.ceil(l * args.fraction))
		train_feats.extend(labeled_instances[:cutoff])
		test_feats.extend(labeled_instances[cutoff:])
		
		if args.trace > 1:
			print '%s: %d training instances, %d testing instances' % (label, cutoff, (l-cutoff))
	else:
		train_feats.extend(labeled_instances)
		test_feats.extend(labeled_instances)

if args.trace:
	print '%d training feats, %d testing feats' % (len(train_feats), len(test_feats))

##################################
## classifier training function ##
##################################

train_kwargs = {}

if args.algorithm == 'DecisionTree':
	train_kwargs['entropy_cutoff'] = args.entropy_cutoff
	train_kwargs['depth_cutoff'] = args.depth_cutoff
	train_kwargs['support_cutoff'] = args.support_cutoff
	train_kwargs['verbose'] = args.trace
elif args.algorithm != 'NaiveBayes':
	if args.algorithm != 'Maxent':
		train_kwargs['algorithm'] = args.algorithm
	
	train_kwargs['max_iter'] = args.max_iter
	train_kwargs['min_ll'] = args.min_ll
	train_kwargs['min_lldelta'] = args.min_lldelta
	train_kwargs['trace'] = args.trace

train_function = {
	'NaiveBayes': NaiveBayesClassifier.train,
	'DecisionTree': DecisionTreeClassifier.train
}

trainf = train_function.get(args.algorithm, MaxentClassifier.train)

if args.trace:
	print 'training a %s classifier' % args.algorithm

classifier = trainf(train_feats, **train_kwargs)

###########################
## classifier evaluation ##
###########################

if not args.no_eval:
	if not args.no_accuracy:
		print 'accuracy: %f' % accuracy(classifier, test_feats)
	
	if not args.no_precision or not args.no_recall or not args.no_fmeasure:
		refsets = collections.defaultdict(set)
		testsets = collections.defaultdict(set)
		
		for i, (feat, label) in enumerate(test_feats):
			refsets[label].add(i)
			observed = classifier.classify(feat)
			testsets[observed].add(i)
		
		for label in labels:
			ref = refsets[label]
			test = testsets[label]
			
			if not args.no_precision:
				print '%s precision: %f' % (label, precision(ref, test))
			
			if not args.no_recall:
				print '%s recall: %f' % (label, recall(ref, test))
			
			if not args.no_fmeasure:
				print '%s f-measure: %f' % (label, f_measure(ref, test))

# TODO: pickle.dump classifier