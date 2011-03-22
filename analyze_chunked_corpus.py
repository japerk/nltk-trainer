#!/usr/bin/python
import argparse
import nltk.corpus
from nltk.tree import Tree
from nltk.corpus.util import LazyCorpusLoader
from nltk.probability import FreqDist
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer import load_corpus_reader

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Analyze a chunked corpus',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a chunked corpus included with NLTK, such as
treebank_chunk or conll2002, or the root path to a corpus directory,
which can be either an absolute path or relative to a nltk_data directory.''')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to %(default)d. 0 is no trace output.')

corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader', default=None,
	help='''Full module path to a corpus reader class, such as
nltk.corpus.reader.chunked.ChunkedCorpusReader''')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')

sort_group = parser.add_argument_group('IOB Count Sorting Options')
sort_group.add_argument('--sort', default='iob', choices=['iob', 'count'],
	help='Sort key, defaults to %(default)s')
sort_group.add_argument('--reverse', action='store_true', default=False,
	help='Sort in revere order')

args = parser.parse_args()

###################
## corpus reader ##
###################

chunked_corpus = load_corpus_reader(args.corpus, reader=args.reader, fileids=args.fileids)

if not chunked_corpus:
	raise ValueError('%s is an unknown corpus')

if args.trace:
	print 'loading nltk.corpus.%s' % args.corpus

##############
## counting ##
##############

wc = 0
iob_counts = FreqDist()
word_set = set()

for obj in chunked_corpus.chunked_words():
	if isinstance(obj, Tree):
		iob_counts.inc(obj.node)
		word_tags = obj.leaves()
	else: # isinstance(obj, tuple)
		word_tags = [obj]
	
	for word, tag in word_tags:
		wc += 1
		word_set.add(word)

############
## output ##
############

print '%d total words\n%d unique words\n%d iob tags\n' % (wc, len(word_set), len(iob_counts))

if args.sort == 'iob':
	sort_key = lambda (t, c): t
elif args.sort == 'count':
	sort_key = lambda (t, c): c
else:
	raise ValueError('%s is not a valid sort option' % args.sort)

# simple reSt table format
print '  IOB      Count  '
print '=======  ========='

for iob, count in sorted(iob_counts.items(), key=sort_key, reverse=args.reverse):
	print '  '.join([iob.ljust(7), str(count).rjust(9)])

print '=======  ========='
