import os, os.path
import cPickle as pickle
import nltk.data
from nltk.tag.simplify import simplify_wsj_tag
from tagging.readers import NumberedTaggedSentCorpusReader

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

def load_corpus_reader(corpus, reader=None, fileids=None):
	if corpus == 'timit':
		return LazyCorpusLoader('timit', NumberedTaggedSentCorpusReader,
			'.+\.tags', tag_mapping_function=simplify_wsj_tag)
	
	real_corpus = getattr(nltk.corpus, corpus, None)
	
	if not real_corpus:
		if not reader:
			raise ValueError('you must specify a corpus reader')
		
		if not fileids:
			raise ValueError('you must specify the corpus fileids')
		
		if os.path.isdir(corpus):
			root = corpus
		else:
			try:
				root = nltk.data.find(corpus)
			except LookupError:
				raise ValueError('cannot find corpus path %s' % corpus)
		
		reader_path, reader_name = reader.rsplit('.', 1)
		mod = __import__(reader_path, globals(), locals(), [reader_name])
		reader_cls = getattr(mod, reader_name)
		# TODO: may also need to support optional args for initialization of reader class
		real_corpus = reader_cls(root, fileids)
	
	return real_corpus