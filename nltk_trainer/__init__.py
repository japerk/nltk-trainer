import os, os.path
import cPickle as pickle
import nltk.data
from nltk.corpus.util import LazyCorpusLoader
from nltk.tag.simplify import simplify_wsj_tag
from nltk_trainer.tagging.readers import NumberedTaggedSentCorpusReader

def dump_object(obj, fname, trace=1):
	dirname = os.path.dirname(fname)
	
	if not os.path.exists(dirname):
		if trace:
			print 'creating directory %s' % dirname
		
		os.mkdir(dirname)
	
	if trace:
		print 'dumping %s to %s' % (obj.__class__.__name__, fname)
	
	f = open(fname, 'wb')
	pickle.dump(obj, f)
	f.close()

def import_attr(path):
	basepath, name = path.rsplit('.', 1)
	mod = __import__(basepath, globals(), locals(), [name])
	return getattr(mod, name)

def load_corpus_reader(corpus, reader=None, fileids=None, **kwargs):
	if corpus == 'timit':
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
		
		reader_cls = import_attr(reader)
		real_corpus = reader_cls(root, fileids, **kwargs)
	
	return real_corpus