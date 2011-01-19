import argparse
import nltk.corpus, nltk.data
from nltk.corpus.util import LazyCorpusLoader
from nltk.probability import ConditionalFreqDist
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer.tagging.readers import NumberedTaggedSentCorpusReader

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Analyze a part-of-speech tagged corpus',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='corpus name/path relative to an nltk_data directory')
parser.add_argument('--tagger',
	help='pickled tagger filename/path relative to an nltk_data directory')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')

corpus_group = parser.add_argument_group('Corpus Options')
# TODO: any plaintext corpus reader should work, or an import path to the reader class
corpus_group.add_argument('--reader', choices=('plaintext', 'tagged'),
	default='plaintext',
	help='specify plaintext or part-of-speech tagged corpus')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''The fraction of the corpus to use for testing coverage''')

args = parser.parse_args()

###################
## corpus reader ##
###################

if args.corpus == 'timit':
	corpus = LazyCorpusLoader('timit', NumberedTaggedSentCorpusReader, '.+\.tags')
elif hasattr(nltk.corpus, args.corpus):
	corpus = getattr(nltk.corpus, args.corpus)

# TODO: use reader to get class, also need to support optional args for initialization of reader class

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
	print 'analyzing tag coverage of %s with %s' % (args.corpus, tagger.__class__.__name__)

tag_word_freqs = ConditionalFreqDist()

for sent in corpus.sents():
	for word, tag in tagger.tag(sent):
		tag_word_freqs[tag].inc(word)

print '  Tag  \t  Count  '
print '=======\t========='
	
for tag in sorted(tag_word_freqs.conditions()):
	print '%s\t%d' % (tag, tag_word_freqs[tag].N())

print '=======\t========='