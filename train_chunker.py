import argparse, math, itertools, os.path
import cPickle as pickle
import nltk.tag, nltk.chunk.util
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
from nltk_trainer.chunking import chunkers
# TODO: readers is shared with train_tagger, so move it elsewhere / separate
from nltk_trainer.tagging import readers

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Train a NLTK Classifier',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a chunked corpus included with NLTK, such as treebank or
conll2000, or the root path to a corpus directory,
which can be either an absolute path or relative to a nltk_data directory.''')
parser.add_argument('--filename',
	help='''filename/path for where to store the pickled tagger.
The default is {corpus}_{algorithm}.pickle in ~/nltk_data/taggers''')
parser.add_argument('--no-pickle', action='store_true', default=False,
	help="Don't pickle and save the tagger")
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to %(default)d. 0 is no trace output.')

corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader', default=None,
	help='''Full module path to a corpus reader class, such as
nltk.corpus.reader.chunked.ChunkedCorpusReader''')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='Fraction of corpus to use for training, defaults to %(default)f')

tagger_group = parser.add_argument_group('Tagger Based Chunker')
tagger_group.add_argument('--sequential', default='ub',
	help='''Sequential Backoff Algorithm. This can be any combination of the following letters:
	u: UnigramTagger
	b: BigramTagger
	t: TrigramTagger
The default is "%(default)s", but you can set this to the empty string
to not train a sequential backoff tagger.''')

classifier_group = parser.add_argument_group('Classifier Based Chunker')
classifier_group.add_argument('--classifier', default=None,
	choices=['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS,
	help='''ClassifierBasedPOSTagger algorithm to use, default is %(default)s.
Maxent uses the default Maxent training algorithm, either CG or iis.''')
classifier_group.add_argument('--cutoff_prob', default=0, type=float,
	help='Cutoff probability for classifier tagger to backoff to previous tagger')

maxent_group = parser.add_argument_group('Maxent Classifier Chunker',
	'These options only apply when a Maxent classifier is chosen.')
maxent_group.add_argument('--max_iter', default=10, type=int,
	help='maximum number of training iterations, defaults to %(default)d')
maxent_group.add_argument('--min_ll', default=0, type=float,
	help='stop classification when average log-likelihood is less than this, default is %(default)d')
maxent_group.add_argument('--min_lldelta', default=0.1, type=float,
	help='''stop classification when the change in average log-likelihood is less than this.
default is %(default)f''')

decisiontree_group = parser.add_argument_group('Decision Tree Classifier Chunker',
	'These options only apply when the DecisionTree classifier is chosen')
decisiontree_group.add_argument('--entropy_cutoff', default=0.05, type=float,
	help='default is 0.05')
decisiontree_group.add_argument('--depth_cutoff', default=100, type=int,
	help='default is 100')
decisiontree_group.add_argument('--support_cutoff', default=10, type=int,
	help='default is 10')

eval_group = parser.add_argument_group('Chunker Evaluation',
	'Evaluation metrics for chunkers')
eval_group.add_argument('--no-eval', action='store_true', default=False,
	help="don't do any evaluation")

args = parser.parse_args()

###################
## corpus reader ##
###################

chunked_corpus = readers.load_corpus_reader(args.corpus, reader=args.reader, fileids=args.fileids)

if not chunked_corpus:
	raise ValueError('%s is an unknown corpus')

if args.trace:
	print 'loading nltk.corpus.%s' % args.corpus
# trigger loading so it has its true class
chunked_corpus.fileids()
fileids = args.fileids
kwargs = {}

if fileids and fileids in chunked_corpus.fileids():
	kwargs['fileids'] = [fileids]

	if args.trace:
		print 'using tagged sentences from %s' % fileids

chunk_trees = chunked_corpus.chunked_sents(**kwargs)

##################
## train chunks ##
##################

nchunks = len(chunk_trees)

if args.fraction == 1.0:
	train_chunks = test_chunks = chunk_trees
else:
	cutoff = int(math.ceil(nchunks * args.fraction))
	train_chunks = chunk_trees[:cutoff]
	test_chunks = chunk_trees[cutoff:]

if args.trace:
	print '%d chunks, training on %d' % (nchunks, len(train_chunks))

##########################
## tagger based chunker ##
##########################

sequential_classes = {
	'u': nltk.tag.UnigramTagger,
	'b': nltk.tag.BigramTagger,
	't': nltk.tag.TrigramTagger
}

if args.sequential:
	tagger_classes = []
	
	for c in args.sequential:
		if c not in sequential_classes:
			raise NotImplementedError('%s is not a valid tagger' % c)
		
		tagger_classes.append(sequential_classes[c])
	
	chunker = chunkers.TagChunker(train_chunks, tagger_classes)

################
## evaluation ##
################

if not args.no_eval:
	print 'evaluating %s' % chunker
	print chunker.evaluate(test_chunks)
	# TODO: more metrics