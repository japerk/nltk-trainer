#!/usr/bin/python
import argparse, collections
import nltk.corpus, nltk.corpus.reader, nltk.data, nltk.tag, nltk.metrics
from nltk.corpus.util import LazyCorpusLoader
from nltk.probability import FreqDist
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer import load_corpus_reader

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Analyze a part-of-speech tagged corpus',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a tagged corpus included with NLTK, such as treebank,
brown, cess_esp, floresta, or the root path to a corpus directory,
which can be either an absolute path or relative to a nltk_data directory.''')
parser.add_argument('--tagger', default=nltk.tag._POS_TAGGER,
	help='''pickled tagger filename/path relative to an nltk_data directory
default is NLTK's default tagger''')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')
parser.add_argument('--metrics', action='store_true', default=False,
	help='Use tagged sentences to determine tagger accuracy and tag precision & recall')

corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader', default=None,
	help='''Full module path to a corpus reader class, such as
nltk.corpus.reader.tagged.TaggedCorpusReader''')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''The fraction of the corpus to use for testing coverage''')

args = parser.parse_args()

###################
## corpus reader ##
###################

corpus = load_corpus_reader(args.corpus, reader=args.reader, fileids=args.fileids)

# TODO: support corpora with alternatives to tagged_sents that work just as well
if args.metrics and not hasattr(corpus, 'tagged_sents'):
	raise ValueError('%s does not support metrics' % args.corpus)

############
## tagger ##
############

if args.trace:
	print 'loading tagger %s' % args.tagger

tagger = nltk.data.load(args.tagger)

#######################
## coverage analysis ##
#######################

if args.trace:
	print 'analyzing tag coverage of %s with %s\n' % (args.corpus, tagger.__class__.__name__)

tags_found = FreqDist()
unknown_words = set()

if args.metrics:
	tags_actual = FreqDist()
	tag_refs = []
	tag_test = []
	tag_word_refs = collections.defaultdict(set)
	tag_word_test = collections.defaultdict(set)
	
	for tagged_sent in corpus.tagged_sents():
		for word, tag in tagged_sent:
			tags_actual.inc(tag)
			tag_refs.append(tag)
			tag_word_refs[tag].add(word)
		
		for word, tag in tagger.tag(nltk.tag.untag(tagged_sent)):
			tags_found.inc(tag)
			tag_test.append(tag)
			tag_word_test[tag].add(word)
			
			if tag == '-NONE-':
				unknown_words.add(word)
	
	print 'Accuracy: %f' % nltk.metrics.accuracy(tag_refs, tag_test)
	print 'Unknown words: %d' % len(unknown_words)
	
	if args.trace:
		print ', '.join(sorted(unknown_words))
	
	print ''
	print '  Tag      Found      Actual      Precision      Recall  '
	print '=======  =========  ==========  =============  =========='
	
	for tag in sorted(set(tags_found.keys()) | set(tags_actual.keys())):
		found = tags_found[tag]
		actual = tags_actual[tag]
		precision = nltk.metrics.precision(tag_word_refs[tag], tag_word_test[tag])
		recall = nltk.metrics.recall(tag_word_refs[tag], tag_word_test[tag])
		print '  '.join([tag.ljust(7), str(found).rjust(9), str(actual).rjust(10),
			str(precision).ljust(13)[:13], str(recall).ljust(10)[:13]])
	
	print '=======  =========  ==========  =============  =========='
else:
	for sent in corpus.sents():
		for word, tag in tagger.tag(sent):
			tags_found.inc(tag)
	
	print '  Tag      Found  '
	print '=======  ========='
	
	for tag in sorted(tags_found.samples()):
		print '  '.join([tag.ljust(7), str(tags_found[tag]).rjust(9)])
	
	print '=======  ========='