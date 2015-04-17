import os, os.path, re, time
import nltk.data
from nltk.corpus.util import LazyCorpusLoader
from nltk_trainer.tagging.readers import NumberedTaggedSentCorpusReader

try:
	import cPickle as pickle
except ImportError:
	import pickle

try:
	from nltk.compat import iteritems
except ImportError:
	def iteritems(d):
		return d.iteritems()

try:
	basestring = basestring
except NameError:
	basestring = unicode = str

try:
	from nltk.tag.simplify import simplify_wsj_tag
except ImportError:
	simplify_wsj_tag = None

def dump_object(obj, fname, trace=1):
	dirname = os.path.dirname(fname)
	
	if dirname and not os.path.exists(dirname):
		if trace:
			print('creating directory %s' % dirname)
		
		os.makedirs(dirname)
	
	if trace:
		print('dumping %s to %s' % (obj.__class__.__name__, fname))
	
	f = open(fname, 'wb')
	pickle.dump(obj, f)
	f.close()

def load_model(path):
	try:
		return nltk.data.load(path)
	except LookupError:
		return pickle.load(open(os.path.expanduser(path)))

def import_attr(path):
	basepath, name = path.rsplit('.', 1)
	mod = __import__(basepath, globals(), locals(), [name])
	return getattr(mod, name)

def load_corpus_reader(corpus, reader=None, fileids=None, sent_tokenizer=None, word_tokenizer=None, **kwargs):
	if corpus == 'timit':
		# TODO: switch to universal
		return LazyCorpusLoader('timit', NumberedTaggedSentCorpusReader,
			'.+\.tags', tag_mapping_function=simplify_wsj_tag)
	
	real_corpus = getattr(nltk.corpus, corpus, None)
	
	if not real_corpus:
		if not reader:
			raise ValueError('you must specify a corpus reader')
		
		if not fileids:
			fileids = '.*'
		
		root = os.path.expanduser(corpus)
		
		if not os.path.isdir(root):
			if not corpus.startswith('corpora/'):
				path = 'corpora/%s' % corpus
			else:
				path = corpus
			
			try:
				root = nltk.data.find(path)
			except LookupError:
				raise ValueError('cannot find corpus path for %s' % corpus)
		
		if sent_tokenizer and isinstance(sent_tokenizer, basestring):
			kwargs['sent_tokenizer'] = nltk.data.load(sent_tokenizer)
		
		if word_tokenizer and isinstance(word_tokenizer, basestring):
			kwargs['word_tokenizer'] = import_attr(word_tokenizer)()
		
		reader_cls = import_attr(reader)
		real_corpus = reader_cls(root, fileids, **kwargs)
	
	return real_corpus

# the major punct this doesn't handle are '"- but that's probably fine
spacepunct_re = re.compile(r'\s([%s])' % re.escape('!.,;:%?)}]'))
punctspace_re = re.compile(r'([%s])\s' % re.escape('{([#$'))

def join_words(words):
	'''
	>>> join_words(['Hello', ',', 'my', 'name', 'is', '.'])
	'Hello, my name is.'
	>>> join_words(['A', 'test', '(', 'for', 'parens', ')', '!'])
	'A test (for parens)!'
	'''
	return punctspace_re.sub(r'\1', spacepunct_re.sub(r'\1', ' '.join(words)))

if __name__ == '__main__':
	import doctest
	doctest.testmod()