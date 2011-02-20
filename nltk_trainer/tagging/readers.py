from nltk.corpus.reader import TaggedCorpusReader

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