#!/usr/bin/env python
import argparse, collections, itertools, operator, re, string, time
import cPickle as pickle
import nltk.data
from nltk.classify.util import accuracy
from nltk.corpus import stopwords
from nltk.metrics import f_measure, precision, recall
from nltk.util import ngrams
from nltk_trainer import load_corpus_reader
from nltk_trainer.classification import corpus, scoring
from nltk_trainer.classification.featx import bag_of_words

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Analyze a classifier on a classified corpus')
parser.add_argument('corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('--classifier', required=True,
	help='pickled classifier name/path relative to an nltk_data directory')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')
parser.add_argument('--metrics', action='store_true', default=False,
	help='Use classified instances to determine classifier accuracy, precision & recall')
parser.add_argument('--speed', action='store_true', default=False,
	help='Determine average instance classification speed.')

corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader',
	default='nltk.corpus.reader.CategorizedPlaintextCorpusReader',
	help='Full module path to a corpus reader class, such as %(default)s')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--cat_pattern', default='(.+)/.+',
	help='''A regular expression pattern to identify categories based on file paths.
	If cat_file is also given, this pattern is used to identify corpus file ids.
	The default is '(.+)/+', which uses sub-directories as categories.''')
corpus_group.add_argument('--cat_file',
	help='relative path to a file containing category listings')
corpus_group.add_argument('--delimiter', default=' ',
	help='category delimiter for category file, defaults to space')
corpus_group.add_argument('--instances', default='paras',
	choices=('sents', 'paras', 'files'),
	help='''the group of words that represents a single training instance,
	the default is to use entire files''')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''The fraction of the corpus to use for testing coverage''')

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

args = parser.parse_args()

###################
## corpus reader ##
###################

reader_args = []
reader_kwargs = {}

if args.cat_pattern:
	reader_args.append(args.cat_pattern)
	reader_kwargs['cat_pattern'] = re.compile(args.cat_pattern)

if args.cat_file:
	reader_kwargs['cat_file'] = args.cat_file
	
	if args.delimiter:
		reader_kwargs['delimiter'] = args.delimiter

categorized_corpus = load_corpus_reader(args.corpus, args.reader, *reader_args, **reader_kwargs)

if args.metrics and not hasattr(categorized_corpus, 'categories'):
	raise ValueError('%s does not support metrics' % args.corpus)

labels = categorized_corpus.categories()

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

	if args.ngrams:
		return reduce(operator.add, [words if n == 1 else ngrams(words, n) for n in args.ngrams])
	else:
		return words

#####################
## text extraction ##
#####################

if args.speed:
	load_start = time.time()

try:
	classifier = nltk.data.load(args.classifier)
except LookupError:
	classifier = pickle.load(open(args.classifier))

if args.speed:
	load_secs = time.time() - load_start
	print 'loading time: %dsecs' % load_secs

if args.metrics:
	label_instance_function = {
		'sents': corpus.category_sent_words,
		'paras': corpus.category_para_words,
		'files': corpus.category_file_words
	}
	
	lif = label_instance_function[args.instances]
	feats = []
	test_feats = []
	
	for label in labels:
		texts = lif(categorized_corpus, label)
		
		if args.instances == 'files':
			# don't get list(texts) here since might have tons of files
			stop = int(len(categorized_corpus.fileids())*args.fraction)
		else:
			texts = list(texts)
			stop = int(len(texts)*args.fraction)
		
		for t in itertools.islice(texts, stop):
			feat = bag_of_words(norm_words(t))
			feats.append(feat)
			test_feats.append((feat, label))
	
	print 'accuracy:', accuracy(classifier, test_feats)
	refsets, testsets = scoring.ref_test_sets(classifier, test_feats)
	
	for label in labels:
		ref = refsets[label]
		test = testsets[label]
		print '%s precision: %f' % (label, precision(ref, test) or 0)
		print '%s recall: %f' % (label, recall(ref, test) or 0)
		print '%s f-measure: %f' % (label, f_measure(ref, test) or 0)
else:
	if args.instances == 'sents':
		texts = categorized_corpus.sents()
		total = len(texts)
	elif args.instances == 'paras':
		texts = (itertools.chain(*para) for para in categorized_corpus.paras())
		total = len(categorized_corpus.paras())
	elif args.instances == 'files':
		texts = (categorized_corpus.words(fileids=[fid]) for fid in categorized_corpus.fileids())
		total = len(categorized_corpus.fileids())
	
	stop = int(total * args.fraction)
	feats = (bag_of_words(norm_words(i)) for i in itertools.islice(texts, stop))

label_counts = collections.defaultdict(int)

if args.speed:
	time_start = time.time()

for feat in feats:
	label = classifier.classify(feat)
	label_counts[label] += 1

if args.speed:
	time_end = time.time()

for label in sorted(label_counts.keys()):
	print label, label_counts[label]

if args.speed:
	secs = (time_end - time_start)
	nfeats = sum(label_counts.values())
	print 'average time per classify: %dsecs / %d feats = %f ms/feat' % (secs, nfeats, (float(secs) / nfeats) * 1000)
