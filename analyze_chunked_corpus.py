#!/usr/bin/env python
import argparse, collections
import nltk.corpus
from nltk.tree import Tree
from nltk.corpus.util import LazyCorpusLoader
from nltk_trainer import load_corpus_reader, simplify_wsj_tag
from nltk_trainer.chunking.transforms import node_label

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

if simplify_wsj_tag:
	corpus_group.add_argument('--simplify_tags', action='store_true', default=False,
		help='Use simplified tags')

sort_group = parser.add_argument_group('Tag Count Sorting Options')
sort_group.add_argument('--sort', default='tag', choices=['tag', 'count'],
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
	print('loading %s' % args.corpus)

##############
## counting ##
##############

wc = 0
tag_counts = collections.defaultdict(int)
iob_counts = collections.defaultdict(int)
tag_iob_counts = collections.defaultdict(lambda: collections.defaultdict(int))
word_set = set()

for obj in chunked_corpus.chunked_words():
	if isinstance(obj, Tree):
		label = node_label(obj)
		iob_counts[label] += 1
		
		for word, tag in obj.leaves():
			wc += 1
			word_set.add(word)
			tag_counts[tag] += 1
			tag_iob_counts[tag][label] += 1
	else:
		word, tag = obj
		wc += 1
		word_set.add(word)
		tag_counts[tag] += 1

############
## output ##
############

print('%d total words' % wc)
print('%d unique words' % len(word_set))
print('%d tags' % len(tag_counts))
print('%d IOBs\n' % len(iob_counts))

if args.sort == 'tag':
	sort_key = lambda tc: tc[0]
elif args.sort == 'count':
	sort_key = lambda tc: tc[1]
else:
	raise ValueError('%s is not a valid sort option' % args.sort)

line1 = '  Tag      Count  '
line2 = '=======  ========='

iobs = sorted(iob_counts.keys())

for iob in iobs:
	line1 += '    %s  ' % iob
	line2 += '  ==%s==' % ('=' * len(iob))

print(line1)
print(line2)

for tag, count in sorted(tag_counts.items(), key=sort_key, reverse=args.reverse):
	iob_counts = [str(tag_iob_counts[tag][iob]).rjust(4+len(iob)) for iob in iobs]
	print('  '.join([tag.ljust(7), str(count).rjust(9)] + iob_counts))

print(line2)