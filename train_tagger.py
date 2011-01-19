import argparse, math, itertools, os.path
import cPickle as pickle
import nltk.corpus
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
# special case corpus readers
from nltk.corpus.reader import SwitchboardCorpusReader, NPSChatCorpusReader, IndianCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from nltk.tag import ClassifierBasedPOSTagger
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer.tagging.readers import NumberedTaggedSentCorpusReader
from nltk_trainer.tagging.training import train_brill_tagger
from nltk_trainer.tagging.taggers import PhoneticClassifierBasedPOSTagger

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Train a NLTK Classifier',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a tagged corpus included with NLTK, such as treebank,
brown, cess_esp, or floresta''')
parser.add_argument('--filename',
	help='''filename/path for where to store the pickled tagger.
The default is {corpus}_{algorithm}.pickle in ~/nltk_data/taggers''')
parser.add_argument('--no-pickle', action='store_true', default=False,
	help="Don't pickle and save the tagger")
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to %(default)d. 0 is no trace output.')
parser.add_argument('--fraction', default=1.0, type=float,
	help='Fraction of corpus to use for training, defaults to %(default)f')
parser.add_argument('--fileid', default=None,
	help='Specify and individual fileid to use for training')

tagger_group = parser.add_argument_group('Tagger Choices')
tagger_group.add_argument('--default', default='-None-',
	help='''The default tag "%(default)s". Set this to a different tag, such as "NN",
to change the default tag.''')
tagger_group.add_argument('--simplify_tags', action='store_true', default=False,
	help='Use simplified tags')

sequential_group = parser.add_argument_group('Sequential Tagger')
sequential_group.add_argument('--sequential', default='aubt',
	help='''Sequential Backoff Algorithm. This can be any combination of the following letters:
	a: AffixTagger
	u: UnigramTagger
	b: BigramTagger
	t: TrigramTagger
The default is "%(default)s", but you can set this to the empty string
to not train a sequential backoff tagger.''')
sequential_group.add_argument('--affix', action='append', type=int,
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
classifier_group.add_argument('--classifier', default=None,
	choices=['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS,
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

maxent_group = parser.add_argument_group('Maxent Classifier Tagger',
	'These options only apply when a Maxent classifier is chosen.')
maxent_group.add_argument('--max_iter', default=10, type=int,
	help='maximum number of training iterations, defaults to %(default)d')
maxent_group.add_argument('--min_ll', default=0, type=float,
	help='stop classification when average log-likelihood is less than this, default is %(default)d')
maxent_group.add_argument('--min_lldelta', default=0.1, type=float,
	help='''stop classification when the change in average log-likelihood is less than this.
default is %(default)f''')

decisiontree_group = parser.add_argument_group('Decision Tree Classifier Tagger',
	'These options only apply when the DecisionTree classifier is chosen')
decisiontree_group.add_argument('--entropy_cutoff', default=0.05, type=float,
	help='default is 0.05')
decisiontree_group.add_argument('--depth_cutoff', default=100, type=int,
	help='default is 100')
decisiontree_group.add_argument('--support_cutoff', default=10, type=int,
	help='default is 10')

eval_group = parser.add_argument_group('Tagger Evaluation',
	'Evaluation metrics for part-of-speech taggers')
eval_group.add_argument('--no-eval', action='store_true', default=False,
	help="don't do any evaluation")
# TODO: word coverage of test words, how many get a tag != '-NONE-'

args = parser.parse_args()

###################
## corpus reader ##
###################

if args.corpus == 'timit':
	tagged_corpus = LazyCorpusLoader('timit', NumberedTaggedSentCorpusReader, '.+\.tags')
else:
	tagged_corpus = getattr(nltk.corpus, args.corpus)

if not tagged_corpus:
	raise ValueError('%s is an unknown corpus')

if args.trace:
	print 'loading nltk.corpus.%s' % args.corpus
# trigger loading so it has its true class
tagged_corpus.fileids()
# fileid is used for corpus naming, if it exists
fileid = args.fileid
kwargs = {}

if args.fileid:
	kwargs['fileids'] = [args.fileid]
# all other corpora are assumed to support simplify_tags kwarg
if args.simplify_tags and args.corpus not in ['conll2000', 'switchboard', 'pl196x']:
	kwargs['simplify_tags'] = True
# these corpora do not support simplify_tags, and have no known workaround
elif args.simplify_tags and args.corpus in ['pl196x']:
	raise ValueError('%s does not support simplify_tags' % args.corpus)

if isinstance(tagged_corpus, SwitchboardCorpusReader):
	if args.fileid:
		raise ValueError('fileid cannot be used with switchboard')
	
	tagged_sents = list(itertools.chain(*[[list(s) for s in d if s] for d in tagged_corpus.tagged_discourses(**kwargs)]))
elif isinstance(tagged_corpus, NPSChatCorpusReader):
	tagged_sents = tagged_corpus.tagged_posts(**kwargs)
elif isinstance(tagged_corpus, IndianCorpusReader):
	if not kwargs.get('fileids'):
		fileid = 'hindi.pos'
		kwargs['fileids'] = [fileid]
	
	tagged_sents = tagged_corpus.tagged_sents(**kwargs)
else:
	tagged_sents = tagged_corpus.tagged_sents(**kwargs)
# manual simplification is needed for these corpora
if args.simplify_tags and args.corpus in ['conll2000', 'switchboard']:
	tagged_sents = [[(word, simplify_wsj_tag(tag)) for (word, tag) in sent] for sent in tagged_sents]

# TODO: support generic tagged corpus readers: TaggedCorpusReader,
# BracketParseCorpusReader, ConllChunkCorpusReader

##################
## tagged sents ##
##################

# can't trust corpus to provide valid sents (indian)
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

classifier_train_kwargs = {}

if args.classifier == 'DecisionTree':
	classifier_train = DecisionTreeClassifier.train
	classifier_train_kwargs['binary'] = False
	classifier_train_kwargs['entropy_cutoff'] = args.entropy_cutoff
	classifier_train_kwargs['depth_cutoff'] = args.depth_cutoff
	classifier_train_kwargs['support_cutoff'] = args.support_cutoff
	classifier_train_kwargs['verbose'] = args.trace
elif args.classifier == 'NaiveBayes':
	classifier_train = NaiveBayesClassifier.train
elif args.classifier:
	if args.classifier != 'Maxent':
		classifier_train_kwargs['algorithm'] = args.classifier
	
	classifier_train = MaxentClassifier.train
	classifier_train_kwargs['max_iter'] = args.max_iter
	classifier_train_kwargs['min_ll'] = args.min_ll
	classifier_train_kwargs['min_lldelta'] = args.min_lldelta
	classifier_train_kwargs['trace'] = args.trace

if args.classifier:
	def classifier_builder(train_feats):
		return classifier_train(train_feats, **classifier_train_kwargs)
	
	kwargs = {
		'train': train_sents,
		'verbose': args.trace,
		'backoff': tagger,
		'cutoff_prob': args.cutoff_prob,
		'classifier_builder': classifier_builder
	}
	
	phonetic_keys = ['metaphone', 'double_metaphone', 'soundex', 'nysiis', 'caverphone']
	
	if any([getattr(args, key) for key in phonetic_keys]):
		cls = PhoneticClassifierBasedPOSTagger
		
		for key in phonetic_keys:
			kwargs[key] = getattr(args, key)
	else:
		cls = ClassifierBasedPOSTagger
	
	if args.trace:
		print 'training a %s %s' % (args.classifier, cls.__name__)
	
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
	print 'evaluating %s' % tagger
	print 'accuracy: %f' % tagger.evaluate(test_sents)

##############
## pickling ##
##############

if not args.no_pickle:
	if args.filename:
		fname = os.path.expanduser(args.filename)
	else:
		parts = [args.corpus]
		
		if fileid:
			parts.append(os.path.splitext(fileid)[0])
		
		if args.brill:
			parts.append('brill')
		
		if args.classifier:
			parts.append(args.classifier)
		
		if args.sequential:
			parts.append(args.sequential)
		
		name = '%s.pickle' % '_'.join(parts)
		fname = os.path.join(os.path.expanduser('~/nltk_data/taggers'), name)
	
	dirname = os.path.dirname(fname)
	
	if not os.path.exists(dirname):
		if args.trace:
			print 'creating directory %s' % dirname
		
		os.mkdir(dirname)
	
	if args.trace:
		print 'dumping tagger to %s' % fname
	
	f = open(fname, 'wb')
	pickle.dump(tagger, f)
	f.close()