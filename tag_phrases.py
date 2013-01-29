#!/usr/bin/env python
import argparse, os.path
import cPickle as pickle
import nltk.data, nltk.tag
from nltk_trainer import load_corpus_reader
from nltk_trainer.writer.chunked import ChunkedCorpusWriter

########################################
## command options & argument parsing ##
########################################

# TODO: many of the args are shared with analyze_classifier_coverage, so abstract

parser = argparse.ArgumentParser(description='Classify a plaintext corpus to a classified corpus')
# TODO: make sure source_corpus can be a single file
parser.add_argument('source_corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('target_corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')
parser.add_argument('--tagger', default=nltk.tag._POS_TAGGER,
	help='''pickled tagger filename/path relative to an nltk_data directory
default is NLTK's default tagger''')

# TODO: from analyze_tagged_corpus.py
corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader',
	default='nltk.corpus.reader.plaintext.PlaintextCorpusReader',
	help='Full module path to a corpus reader class, defaults to %(default)s.')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--sent-tokenizer', default='tokenizers/punkt/english.pickle',
	help='Path to pickled sentence tokenizer')
corpus_group.add_argument('--word-tokenizer', default='nltk.tokenize.WordPunctTokenizer',
	help='Full module path to a tokenizer class, defaults to %(default)s.')

args = parser.parse_args()

###################
## corpus reader ##
###################

source_corpus = load_corpus_reader(args.source_corpus, reader=args.reader,
	fileids=args.fileids, encoding='utf-8', sent_tokenizer=args.sent_tokenizer,
	word_tokenizer=args.word_tokenizer)

if not source_corpus:
	raise ValueError('%s is an unknown corpus')

if args.trace:
	print 'loaded %s' % args.source_corpus

############
## tagger ##
############

# TODO: from analyze_tagger_coverage.py
if args.trace:
	print 'loading tagger %s' % args.tagger

try:
	tagger = nltk.data.load(args.tagger)
except LookupError:
	try:
		import cPickle as pickle
	except ImportError:
		import pickle
	
	tagger = pickle.load(open(os.path.expanduser(args.tagger)))

#############
## tagging ##
#############

with ChunkedCorpusWriter(fileids=source_corpus.fileids(), path=args.target_corpus) as writer:
	for fileid in source_corpus.fileids():
		paras = source_corpus.paras(fileids=[fileid])
		tagged_paras = ((tagger.tag(sent) for sent in para) for para in paras)
		writer.write_paras(tagged_paras, fileid=fileid)