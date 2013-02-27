#!/usr/bin/env python
import argparse, collections, itertools, math, os.path, re, string, operator
import nltk.data
import nltk_trainer.classification.args
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import stopwords
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, CategorizedTaggedCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from nltk.metrics import BigramAssocMeasures, f_measure, masi_distance, precision, recall
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.util import ngrams
from nltk_trainer import dump_object, import_attr, load_corpus_reader
from nltk_trainer.classification import corpus, scoring
from nltk_trainer.classification.featx import (bag_of_words, bag_of_words_in_set,
	word_counts, train_test_feats)
from nltk_trainer.classification.multi import MultiBinaryClassifier

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Train a NLTK Classifier')

parser.add_argument('corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('--filename', help='''filename/path for where to store the
	pickled classifier, the default is {corpus}_{algorithm}.pickle in
	~/nltk_data/classifiers''')
parser.add_argument('--no-pickle', action='store_true', default=False,
	help="don't pickle and save the classifier")
parser.add_argument('--classifier', '--algorithm', default=['NaiveBayes'], nargs='+',
	choices=nltk_trainer.classification.args.classifier_choices,
	help='''Classifier algorithm to use, defaults to %(default)s. Maxent uses the
	default Maxent training algorithm, either CG or iis.''')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')
parser.add_argument('--show-most-informative', default=0, type=int,
	help='number of most informative features to show, works for all algorithms except DecisionTree')

corpus_group = parser.add_argument_group('Training Corpus')
corpus_group.add_argument('--reader',
	default='nltk.corpus.reader.CategorizedPlaintextCorpusReader',
	help='Full module path to a corpus reader class, such as %(default)s')
corpus_group.add_argument('--cat_pattern', default='(.+)/.+',
	help='''A regular expression pattern to identify categories based on file paths.
	If cat_file is also given, this pattern is used to identify corpus file ids.
	The default is '(.+)/+', which uses sub-directories as categories.''')
corpus_group.add_argument('--cat_file',
	help='relative path to a file containing category listings')
corpus_group.add_argument('--delimiter', default=' ',
	help='category delimiter for category file, defaults to space')
corpus_group.add_argument('--instances', default='files',
	choices=('sents', 'paras', 'files'),
	help='''the group of words that represents a single training instance,
	the default is to use entire files''')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''The fraction of the corpus to use for training a binary or
	multi-class classifier, the rest will be used for evaulation.
	The default is to use the entire corpus, and to test the classifier
	against the same training data. Any number < 1 will test against
	the remaining fraction.''')
corpus_group.add_argument('--train-prefix', default=None,
	help='optional training fileid prefix for multi classifiers')
corpus_group.add_argument('--test-prefix', default=None,
	help='optional testing fileid prefix for multi classifiers')
corpus_group.add_argument('--word-tokenizer', default='', help='Word Tokenizer class path')
corpus_group.add_argument('--sent-tokenizer', default='', help='Sent Tokenizer data.pickle path')
corpus_group.add_argument('--para-block-reader', default='', help='Block reader function path')

classifier_group = parser.add_argument_group('Classifier Type',
	'''A binary classifier has only 2 labels, and is the default classifier type.
	A multi-class classifier chooses one of many possible labels.
	A multi-binary classifier choose zero or more labels by combining multiple
	binary classifiers, 1 for each label.''')
classifier_group.add_argument('--binary', action='store_true', default=False,
	help='train a binary classifier, or a multi-binary classifier if --multi is also given')
classifier_group.add_argument('--multi', action='store_true', default=False,
	help='train a multi-class classifier, or a multi-binary classifier if --binary is also given')

feat_group = parser.add_argument_group('Feature Extraction',
	'The default is to lowercase every word, strip punctuation, and use stopwords')
feat_group.add_argument('--ngrams', nargs='+', type=int,
	help='use n-grams as features.')
feat_group.add_argument('--no-lowercase', action='store_true', default=False,
	help="don't lowercase every word")
feat_group.add_argument('--filter-stopwords', default='no',
	choices=['no']+stopwords.fileids(),
	help='language stopwords to filter, defaults to "no" to keep stopwords')
feat_group.add_argument('--punctuation', action='store_true', default=False,
	help="don't strip punctuation")
feat_group.add_argument('--value-type', default='bool', choices=('bool', 'int', 'float'),
	help='''Data type of values in featuresets. The default is bool, which ignores word counts.
	Use int to get word and/or ngram counts.''')

score_group = parser.add_argument_group('Feature Scoring',
	'The default is no scoring, all words are included as features')
score_group.add_argument('--score_fn', default='chi_sq',
	choices=[f for f in dir(BigramAssocMeasures) if not f.startswith('_')],
	help='scoring function for information gain and bigram collocations, defaults to chi_sq')
score_group.add_argument('--min_score', default=0, type=int,
	help='minimum score for a word to be included, default is 0 to include all words')
score_group.add_argument('--max_feats', default=0, type=int,
	help='maximum number of words to include, ordered by highest score, defaults is 0 to include all words')

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
eval_group.add_argument('--cross-fold', type=int, default=0,
	help='''If given a number greater than 2, will do cross fold validation
	instead of normal training and testing. This option implies --no-pickle,
	is useless with --trace 0 and/or --no-eval, and currently does not work
	with --multi --binary.
	''')

nltk_trainer.classification.args.add_maxent_args(parser)
nltk_trainer.classification.args.add_decision_tree_args(parser)
nltk_trainer.classification.args.add_sklearn_args(parser)

args = parser.parse_args()

###################
## corpus reader ##
###################

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

if args.word_tokenizer:
	reader_kwargs['word_tokenizer'] = import_attr(args.word_tokenizer)()

if args.sent_tokenizer:
	reader_kwargs['sent_tokenizer'] = nltk.data.LazyLoader(args.sent_tokenizer)

if args.para_block_reader:
	reader_kwargs['para_block_reader'] = import_attr(args.para_block_reader)

if args.trace:
	print 'loading %s' % args.corpus

categorized_corpus = load_corpus_reader(args.corpus, args.reader,
	*reader_args, **reader_kwargs)

if not hasattr(categorized_corpus, 'categories'):
	raise ValueError('%s is does not have categories for classification')

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

########################
## text normalization ##
########################

if args.filter_stopwords == 'no':
	stopset = set()
else:
	stopset = set(stopwords.words(args.filter_stopwords))

if not args.punctuation:
	stopset |= set(string.punctuation)

def norm_words(words):
	if not args.no_lowercase:
		words = [w.lower() for w in words]
	
	if not args.punctuation:
		words = [w.strip(string.punctuation) for w in words]
		words = [w for w in words if w]
	
	if stopset:
		words = [w for w in words if w.lower() not in stopset]
	# in case nothing has happened to words, ensure is a list so can add together
	if not isinstance(words, list):
		words = list(words)
	
	if args.ngrams:
		return reduce(operator.add, [words if n == 1 else ngrams(words, n) for n in args.ngrams])
	else:
		return words

##################
## word scoring ##
##################

score_fn = getattr(BigramAssocMeasures, args.score_fn)

if args.min_score or args.max_feats:
	if args.trace:
		print 'calculating word scores'
	
	cat_words = [(cat, norm_words(words)) for cat, words in corpus.category_words(categorized_corpus)]
	ws = scoring.sorted_word_scores(scoring.sum_category_word_scores(cat_words, score_fn))
	
	if args.min_score:
		ws = [(w, s) for (w, s) in ws if s >= args.min_score]
	
	if args.max_feats:
		ws = ws[:args.max_feats]
	
	bestwords = set([w for (w, s) in ws])
	
	if args.value_type == 'bool':
		if args.trace:
			print 'using bag of words from known set feature extraction'
		
		featx = lambda words: bag_of_words_in_set(words, bestwords)
	else:
		if args.trace:
			print 'using word counts from known set feature extraction'
		
		featx = lambda words: word_counts_in_set(words, bestwords)
	
	if args.trace:
		print '%d words meet min_score and/or max_feats' % len(bestwords)
elif args.value_type == 'bool':
	if args.trace:
		print 'using bag of words feature extraction'
	
	featx = bag_of_words
else:
	if args.trace:
		print 'using word counts feature extraction'
	
	featx = word_counts

#####################
## text extraction ##
#####################

if args.multi and args.binary:
	label_instance_function = {
		'sents': corpus.multi_category_sent_words,
		'paras': corpus.multi_category_para_words,
		'files': corpus.multi_category_file_words
	}
	
	lif = label_instance_function[args.instances]
	train_instances = lif(categorized_corpus, args.train_prefix)
	test_instances = lif(categorized_corpus, args.test_prefix)
	train_feats = [(featx(norm_words(words)), cats) for words, cats in train_instances]
	test_feats = [(featx(norm_words(words)), cats) for words, cats in test_instances]
else:
	label_instance_function = {
		'sents': corpus.category_sent_words,
		'paras': corpus.category_para_words,
		'files': corpus.category_file_words
	}
	
	lif = label_instance_function[args.instances]
	label_instances = {}
	
	for label in labels:
		instances = [norm_words(i) for i in lif(categorized_corpus, label)]
		label_instances[label] = [i for i in instances if i]
	
	train_feats = []
	test_feats = []
	
	for label, instances in label_instances.iteritems():
		ltrain_feats, ltest_feats = train_test_feats(label, instances, featx=featx, fraction=args.fraction)
		
		if args.trace > 1:
			info = (label, len(ltrain_feats), len(ltest_feats))
			print '%s: %d training instances, %d testing instances' % info
		
		train_feats.extend(ltrain_feats)
		test_feats.extend(ltest_feats)
	
if args.trace:
	print '%d training feats, %d testing feats' % (len(train_feats), len(test_feats))

##############
## training ##
##############

trainf = nltk_trainer.classification.args.make_classifier_builder(args)

if args.multi and args.binary:
	if args.trace:
		print 'training multi-binary %s classifier' % args.classifier
	
	classifier = MultiBinaryClassifier.train(labels, train_feats, trainf)
elif args.cross_fold:
	scoring.cross_fold(train_feats, trainf, accuracy, folds=args.cross_fold,
		trace=args.trace, metrics=not args.no_eval, informative=args.show_most_informative)
else:
	classifier = trainf(train_feats)

################
## evaluation ##
################

if not args.no_eval and not args.cross_fold:
	if not args.no_accuracy:
		try:
			print 'accuracy: %f' % accuracy(classifier, test_feats)
		except ZeroDivisionError:
			print 'accuracy: 0'
	
	if args.multi and args.binary and not args.no_masi_distance:
		print 'average masi distance: %f' % (scoring.avg_masi_distance(classifier, test_feats))
	
	if not args.no_precision or not args.no_recall or not args.no_fmeasure:
		if args.multi and args.binary:
			refsets, testsets = scoring.multi_ref_test_sets(classifier, test_feats)
		else:
			refsets, testsets = scoring.ref_test_sets(classifier, test_feats)
		
		for label in labels:
			ref = refsets[label]
			test = testsets[label]
			
			if not args.no_precision:
				print '%s precision: %f' % (label, precision(ref, test) or 0)
			
			if not args.no_recall:
				print '%s recall: %f' % (label, recall(ref, test) or 0)
			
			if not args.no_fmeasure:
				print '%s f-measure: %f' % (label, f_measure(ref, test) or 0)

if args.show_most_informative and hasattr(classifier, 'show_most_informative_features') and not (args.multi and args.binary) and not args.cross_fold:
	print '%d most informative features' % args.show_most_informative
	classifier.show_most_informative_features(args.show_most_informative)

##############
## pickling ##
##############

if not args.no_pickle and not args.cross_fold:
	if args.filename:
		fname = os.path.expanduser(args.filename)
	else:
		name = '%s_%s.pickle' % (args.corpus, '_'.join(args.classifier))
		fname = os.path.join(os.path.expanduser('~/nltk_data/classifiers'), name)
	
	dump_object(classifier, fname, trace=args.trace)
