import os.path
import nltk.corpus
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tag.simplify import simplify_wsj_tag

def numbered_sent_block_reader(stream):
	line = stream.readline()
	
	if not line:
		return []
	
	n, sent = line.split(' ', 1)
	return [sent]

class NumberedTaggedSentCorpusReader(TaggedCorpusReader):
	def __init__(self, *args, **kwargs):
		super(NumberedTaggedSentCorpusReader, self).__init__(
			para_block_reader=numbered_sent_block_reader, *args, **kwargs)
	
	def paras(self):
		raise NotImplementedError('use sents()')
	
	def tagged_paras(self):
		raise NotImplementedError('use tagged_sents()')

def load_corpus_reader(corpus, reader=None, fileids=None):
	if corpus == 'timit':
		return LazyCorpusLoader('timit', NumberedTaggedSentCorpusReader,
			'.+\.tags', tag_mapping_function=simplify_wsj_tag)
	
	tagged_corpus = getattr(nltk.corpus, corpus, None)
	
	if not tagged_corpus:
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
		tagged_corpus = reader_cls(root, fileids)
	
	return tagged_corpus