#!/usr/bin/env python
import argparse, itertools, operator, os, os.path, string
import nltk.data
from nltk.corpus import stopwords
from nltk.misc import babelfish
from nltk.tokenize import wordpunct_tokenize
from nltk.util import ngrams
from nltk_trainer import load_corpus_reader, join_words, translate
from nltk_trainer.classification.featx import bag_of_words

langs = [l.lower() for l in babelfish.available_languages]

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

classifier_group = parser.add_argument_group('Classification Options')
parser.add_argument('--classifier', default=None,
	help='pickled classifier name/path relative to an nltk_data directory')
parser.add_argument('--wordlist', default=None,
	help='classified word list corpus for word/phrase classification')
parser.add_argument('--threshold', type=float, default=0.9,
	help='Minimum probability required to write classified instance')

corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader',
	default='nltk.corpus.reader.CategorizedPlaintextCorpusReader',
	help='Full module path to a corpus reader class, such as %(default)s')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--instances', default='paras', choices=('sents', 'paras'),
	help='''the group of words that represents a single training instance,
	the default is to use entire files''')

feat_group = parser.add_argument_group('Feature Extraction',
	'The default is to lowercase every word, strip punctuation, and use stopwords')
feat_group.add_argument('--ngrams', action='append', type=int,
	help='use n-grams as features.')
feat_group.add_argument('--no-lowercase', action='store_true', default=False,
	help="don't lowercase every word")
feat_group.add_argument('--filter-stopwords', default='no',
	choices=['no']+stopwords.fileids(),
	help='language stopwords to filter, defaults to "no" to keep stopwords')
feat_group.add_argument('--punctuation', action='store_true', default=False,
	help="don't strip punctuation")

trans_group = parser.add_argument_group('Language Translation')
trans_group.add_argument('--source', default='english', choices=langs, help='source language')
trans_group.add_argument('--target', default=None, choices=langs, help='target language')
trans_group.add_argument('--retries', default=3, type=int,
	help='Number of babelfish retries before quiting')
trans_group.add_argument('--sleep', default=3, type=int,
	help='Sleep time between retries')

args = parser.parse_args()

###################
## corpus reader ##
###################

source_corpus = load_corpus_reader(args.source_corpus, args.reader)

if not source_corpus:
	raise ValueError('%s is an unknown corpus')

if args.trace:
	print 'loaded %s' % args.source_corpus

########################
## text normalization ##
########################

# TODO: copied from analyze_classifier_coverage, so abstract

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

##############
## classify ##
##############

if args.wordlist:
	classifier = WordListClassifier(load_corpus_reader(args.wordlist))
elif args.classifier:
	if args.trace:
		print 'loading %s' % args.classifier
	
	classifier = nltk.data.load(args.classifier)
else:
	raise ValueError('one of wordlist or classifier is needed')

def label_filename(label):
	# TODO: better file path based on args.target_corpus & label
	path = os.path.join(args.target_corpus, '%s.txt' % label)
	
	if not os.path.exists(args.target_corpus):
		os.makedirs(args.target_corpus)
	
	if args.trace:
		print 'filename for category %s: %s' % (label, path)
	
	return path

labels = classifier.labels()
label_files = dict([(l, open(label_filename(l), 'a')) for l in labels])

# TODO: create a nltk.corpus.writer framework with some initial CorpusWriter classes

if args.target:
	if args.trace:
		print 'translating all text from %s to %s' % (args.source, args.target)
	
	featx = lambda words: bag_of_words(norm_words(wordpunct_tokenize(translate(join_words(words),
		args.source, args.target, trace=args.trace, sleep=args.sleep, retries=args.retries))))
else:
	featx = lambda words: bag_of_words(norm_words(words))

def classify_write(words):
	feats = featx(words)
	probs = classifier.prob_classify(feats)
	label = probs.max()
	
	if probs.prob(label) >= args.threshold:
		label_files[label].write(join_words(words) + u'\n\n')

if args.trace:
	print 'classifying %s' % args.instances

if args.instances == 'paras':
	for para in source_corpus.paras():
		classify_write(list(itertools.chain(*para)))
else: # args.instances == 'sents'
	for sent in source_corpus.sents():
		classify_write(sent)


# TODO: arg(s) to specify categorized word list corpus instead of classifier pickle
# can have additional arguments for decision threshold. this will create a
# KeywordClassifier that can be used just like any other NLTK classifier

# TODO: if new corpus files already exist, append to them, and make sure the
# first append example is separate (enough) from the last example in the file
# (we don't want to append a paragraph right next to another paragraph, creating a single paragraph)