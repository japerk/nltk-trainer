#!/usr/bin/env python
import argparse, math, itertools, os.path
import nltk.corpus, nltk.data
import nltk_trainer.classification.args
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
# special case corpus readers
from nltk.corpus.reader import SwitchboardCorpusReader, NPSChatCorpusReader, IndianCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from nltk.tag import ClassifierBasedPOSTagger
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer import dump_object, load_corpus_reader
from nltk_trainer.tagging import readers
from nltk_trainer.tagging.training import train_brill_tagger
from nltk_trainer.tagging.taggers import PhoneticClassifierBasedPOSTagger

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Train a NLTK Classifier',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a tagged corpus included with NLTK, such as treebank,
brown, cess_esp, floresta, or the root path to a corpus directory,
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
nltk.corpus.reader.tagged.TaggedCorpusReader''')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='Fraction of corpus to use for training, defaults to %(default)f')

tagger_group = parser.add_argument_group('Tagger Choices')
tagger_group.add_argument('--default', default='-None-',
	help='''The default tag "%(default)s". Set this to a different tag, such as "NN",
to change the default tag.''')
tagger_group.add_argument('--simplify_tags', action='store_true', default=False,
	help='Use simplified tags')
tagger_group.add_argument('--backoff', default=None,
	help='Path to pickled backoff tagger. If given, replaces default tagger.')

sequential_group = parser.add_argument_group('Sequential Tagger')
sequential_group.add_argument('--sequential', default='aubt',
	help='''Sequential Backoff Algorithm. This can be any combination of the following letters:
	a: AffixTagger
	u: UnigramTagger
	b: BigramTagger
	t: TrigramTagger
The default is "%(default)s", but you can set this to the empty string
to not train a sequential backoff tagger.''')
sequential_group.add_argument('-a', '--affix', action='append', type=int,
	help='''Add affixes to use for one or more AffixTaggers.
Negative numbers are suffixes, positive numbers are prefixes.
You can use this option multiple times to create multiple AffixTaggers with different affixes.
The affixes will be used in the order given.''')

brill_group = parser.add_argument_group('Brill Tagger Options')
brill_group.add_argument('--brill', action='store_true', default=False,
	help='Train a Brill Tagger in front of the other tagger.')
brill_group.add_argument('--template_bounds', type=int, default=1,
	help='''Choose the max bounds for Brill Templates to train a Brill Tagger.
The default is %(default)d.''')
brill_group.add_argument('--max_rules', type=int, default=200)
brill_group.add_argument('--min_score', type=int, default=2)

classifier_group = parser.add_argument_group('Classifier Based Tagger')
classifier_group.add_argument('--classifier', nargs='*',
	choices=nltk_trainer.classification.args.classifier_choices,
	help='''ClassifierBasedPOSTagger algorithm to use, default is %(default)s.
Maxent uses the default Maxent training algorithm, either CG or iis.''')
classifier_group.add_argument('--cutoff_prob', default=0, type=float,
	help='Cutoff probability for classifier tagger to backoff to previous tagger')

phonetic_group = parser.add_argument_group('Phonetic Feature Options for a Classifier Based Tagger')
phonetic_group.add_argument('--metaphone', action='store_true',
	default=False, help='Use metaphone feature')
phonetic_group.add_argument('--double-metaphone', action='store_true',
	default=False, help='Use double metaphone feature')
phonetic_group.add_argument('--soundex', action='store_true',
	default=False, help='Use soundex feature')
phonetic_group.add_argument('--nysiis', action='store_true',
	default=False, help='Use NYSIIS feature')
phonetic_group.add_argument('--caverphone', action='store_true',
	default=False, help='Use caverphone feature')

nltk_trainer.classification.args.add_maxent_args(parser)
nltk_trainer.classification.args.add_decision_tree_args(parser)

eval_group = parser.add_argument_group('Tagger Evaluation',
	'Evaluation metrics for part-of-speech taggers')
eval_group.add_argument('--no-eval', action='store_true', default=False,
	help="don't do any evaluation")
# TODO: word coverage of test words, how many get a tag != '-NONE-'

args = parser.parse_args()

###################
## corpus reader ##
###################

if args.trace:
	print 'loading %s' % args.corpus

tagged_corpus = load_corpus_reader(args.corpus, reader=args.reader, fileids=args.fileids)
fileids = args.fileids
kwargs = {}

# all other corpora are assumed to support simplify_tags kwarg
if args.simplify_tags and args.corpus not in ['conll2000', 'switchboard', 'pl196x']:
	kwargs['simplify_tags'] = True
# these corpora do not support simplify_tags, and have no known workaround
elif args.simplify_tags and args.corpus in ['pl196x']:
	raise ValueError('%s does not support simplify_tags' % args.corpus)

if isinstance(tagged_corpus, SwitchboardCorpusReader):
	if fileids:
		raise ValueError('fileids cannot be used with switchboard')
	
	tagged_sents = list(itertools.chain(*[[list(s) for s in d if s] for d in tagged_corpus.tagged_discourses(**kwargs)]))
elif isinstance(tagged_corpus, NPSChatCorpusReader):
	tagged_sents = tagged_corpus.tagged_posts(**kwargs)
else:
	if isinstance(tagged_corpus, IndianCorpusReader) and not fileids:
		fileids = 'hindi.pos'
	
	if fileids and fileids in tagged_corpus.fileids():
		kwargs['fileids'] = [fileids]
	
		if args.trace:
			print 'using tagged sentences from %s' % fileids
	
	tagged_sents = tagged_corpus.tagged_sents(**kwargs)

# manual simplification is needed for these corpora
if args.simplify_tags and args.corpus in ['conll2000', 'switchboard']:
	tagged_sents = [[(word, simplify_wsj_tag(tag)) for (word, tag) in sent] for sent in tagged_sents]

##################
## tagged sents ##
##################

# can't trust corpus to provide valid list of sents (indian)
tagged_sents = [sent for sent in tagged_sents if sent]
nsents = len(tagged_sents)

if args.fraction == 1.0:
	train_sents = test_sents = tagged_sents
else:
	cutoff = int(math.ceil(nsents * args.fraction))
	train_sents = tagged_sents[:cutoff]
	test_sents = tagged_sents[cutoff:]

if args.trace:
	print '%d tagged sents, training on %d' % (nsents, len(train_sents))

####################
## default tagger ##
####################

if args.backoff:
	if args.trace:
		print 'loading backoff tagger %s' % args.backoff
	
	tagger = nltk.data.load(args.backoff)
else:
	tagger = nltk.tag.DefaultTagger(args.default)

################################
## sequential backoff taggers ##
################################

# NOTE: passing in verbose=args.trace doesn't produce useful printouts

def affix_constructor(train_sents, backoff=None):
	affixes = args.affix or [-3]
	
	for affix in affixes:
		if args.trace:
			print 'training AffixTagger with affix %d and backoff %s' % (affix, backoff)
		
		backoff = nltk.tag.AffixTagger(train_sents, affix_length=affix,
			min_stem_length=min(affix, 2), backoff=backoff)
	
	return backoff

def ngram_constructor(cls):
	def f(train_sents, backoff=None):
		if args.trace:
			print 'training %s tagger with backoff %s' % (cls, backoff)
		# TODO: args.cutoff option
		return cls(train_sents, backoff=backoff)
	
	return f

sequential_constructors = {
	'a': affix_constructor,
	'u': ngram_constructor(nltk.tag.UnigramTagger),
	'b': ngram_constructor(nltk.tag.BigramTagger),
	't': ngram_constructor(nltk.tag.TrigramTagger)
}

if args.sequential:
	for c in args.sequential:
		if c not in sequential_constructors:
			raise NotImplementedError('%s is not a valid sequential backoff tagger' % c)
		
		constructor = sequential_constructors[c]
		tagger = constructor(train_sents, backoff=tagger)

#######################
## classifier tagger ##
#######################

if args.classifier:
	kwargs = {
		'train': train_sents,
		'verbose': args.trace,
		'backoff': tagger,
		'cutoff_prob': args.cutoff_prob,
		'classifier_builder': nltk_trainer.classification.args.make_classifier_builder(args)
	}
	
	phonetic_keys = ['metaphone', 'double_metaphone', 'soundex', 'nysiis', 'caverphone']
	
	if any([getattr(args, key) for key in phonetic_keys]):
		cls = PhoneticClassifierBasedPOSTagger
		
		for key in phonetic_keys:
			kwargs[key] = getattr(args, key)
	else:
		cls = ClassifierBasedPOSTagger
	
	if args.trace:
		print 'training %s %s' % (args.classifier, cls.__name__)
	
	tagger = cls(**kwargs)

##################
## brill tagger ##
##################

if args.brill:
	tagger = train_brill_tagger(tagger, train_sents, args.template_bounds,
		trace=args.trace, max_rules=args.max_rules, min_score=args.min_score)

################
## evaluation ##
################

if not args.no_eval:
	print 'evaluating %s' % tagger.__class__.__name__
	print 'accuracy: %f' % tagger.evaluate(test_sents)

##############
## pickling ##
##############

if not args.no_pickle:
	if args.filename:
		fname = os.path.expanduser(args.filename)
	else:
		# use the last part of the corpus name/path as the prefix
		parts = [os.path.split(args.corpus.rstrip('/'))[-1]]
		
		if args.brill:
			parts.append('brill')
		
		if args.classifier:
			parts.append('_'.join(args.classifier))
		
		if args.sequential:
			parts.append(args.sequential)
		
		name = '%s.pickle' % '_'.join(parts)
		fname = os.path.join(os.path.expanduser('~/nltk_data/taggers'), name)
	
	dump_object(tagger, fname, trace=args.trace)