import argparse, math, itertools, os.path
import nltk.tag, nltk.chunk.util
import nltk_trainer.classification.args
from nltk_trainer import dump_object
from nltk_trainer.chunking import chunkers
# TODO: readers is shared with train_tagger, so move it elsewhere / separate
from nltk_trainer.tagging import readers

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Train a NLTK Classifier',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a chunked corpus included with NLTK, such as treebank_chunk or
conll2000, or the root path to a corpus directory, which can be either an
absolute path or relative to a nltk_data directory.''')
parser.add_argument('--filename',
	help='''filename/path for where to store the pickled tagger.
The default is {corpus}_{algorithm}.pickle in ~/nltk_data/chunkers''')
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

chunker_group = parser.add_argument_group('Chunker Options')
chunker_group.add_argument('--sequential', default='ub',
	help='''Sequential Backoff Algorithm for a Tagger based Chunker.
This can be any combination of the following letters:
	u: UnigramTagger
	b: BigramTagger
	t: TrigramTagger
The default is "%(default)s". If you specify a classifier, this option will be ignored.''')
chunker_group.add_argument('--classifier', default=None,
	choices=nltk_trainer.classification.args.classifier_choices,
	help='''ClassifierChunker algorithm to use instead of a sequential Tagger based Chunker.
Maxent uses the default Maxent training algorithm, either CG or iis.''')

nltk_trainer.classification.args.add_maxent_args(parser)
nltk_trainer.classification.args.add_decision_tree_args(parser)

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

if not hasattr(chunked_corpus, 'chunked_sents'):
	raise ValueError('%s does not have chunked sents' % args.corpus)

if fileids and fileids in chunked_corpus.fileids():
	kwargs['fileids'] = [fileids]

	if args.trace:
		print 'using chunked sentences from %s' % fileids

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

if args.sequential and not args.classifier:
	tagger_classes = []
	
	for c in args.sequential:
		if c not in sequential_classes:
			raise NotImplementedError('%s is not a valid tagger' % c)
		
		tagger_classes.append(sequential_classes[c])
	
	chunker = chunkers.TagChunker(train_chunks, tagger_classes)

##############################
## classifier based chunker ##
##############################

if args.classifier:
	if args.trace:
		print 'training a %s ClassifierChunker' % args.classifier
	
	chunker = chunkers.ClassifierChunker(train_chunks, verbose=args.trace,
		classifier_builder=nltk_trainer.classification.args.make_classifier_builder(args))

################
## evaluation ##
################

if not args.no_eval:
	print 'evaluating %s' % chunker.__class__.__name__
	print chunker.evaluate(test_chunks)

##############
## pickling ##
##############

if not args.no_pickle:
	if args.filename:
		fname = os.path.expanduser(args.filename)
	else:
		# use the last part of the corpus name/path as the prefix
		parts = [os.path.split(args.corpus.rstrip('/'))[-1]]
		
		if args.classifier:
			parts.append(args.classifier)
		elif args.sequential:
			parts.append(args.sequential)
		
		name = '%s.pickle' % '_'.join(parts)
		fname = os.path.join(os.path.expanduser('~/nltk_data/chunkers'), name)
	
	dump_object(chunker, fname, trace=args.trace)