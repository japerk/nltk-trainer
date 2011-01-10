import argparse
import nltk.corpus
from nltk.corpus.util import LazyCorpusLoader
from nltk.probability import FreqDist
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer.tagging.readers import NumberedTaggedSentCorpusReader

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Analyze a part-of-speech tagged corpus',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a tagged corpus included with NLTK, such as treebank,
brown, cess_esp, or floresta''')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to %(default)d. 0 is no trace output.')
parser.add_argument('--simplify_tags', action='store_true', default=False,
	help='Use simplified tags')
parser.add_argument('--sort', default='tag', choices=['tag', 'count'],
	help='Sort key, defaults to %(default)s')
parser.add_argument('--reverse', action='store_true', default=False,
	help='Sort in revere order')

args = parser.parse_args()

###################
## corpus reader ##
###################

if args.corpus == 'timit':
	tagged_corpus = LazyCorpusLoader('timit', NumberedTaggedSentCorpusReader,
		'.+\.tags', tag_mapping_function=simplify_wsj_tag)
else:
	tagged_corpus = getattr(nltk.corpus, args.corpus)

if not tagged_corpus:
	raise ValueError('%s is an unknown corpus')

if args.trace:
	print 'loading nltk.corpus.%s' % args.corpus

##############
## counting ##
##############

wc = 0
tag_counts = FreqDist()

if args.corpus in ['conll2000', 'switchboard']:
	# TODO: override to support simplified tags
	if args.simplify_tags:
		raise ValueError('%s does not support simplified tags' % args.corpus)
	
	kwargs = {}
else:
	kwargs = {'simplify_tags': args.simplify_tags}

for word, tag in tagged_corpus.tagged_words(**kwargs):
	wc += 1
	tag_counts.inc(tag)

############
## output ##
############

print '%d words\n%d tags\n' % (wc, len(tag_counts))

if args.sort == 'tag':
	sort_key = lambda (t, c): t
elif args.sort == 'count':
	sort_key = lambda (t, c): c
else:
	raise ValueError('%s is not a valid sort option' % args.sort)

# simple reSt table format
print '  Tag  \t  Count  '
print '=======\t========='

for tag, count in sorted(tag_counts.items(), key=sort_key, reverse=args.reverse):
	print '%s\t%s' % (tag, count)

print '=======\t========='