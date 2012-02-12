from nltk.tag.util import tuple2str
from nltk_trainer.writer import CorpusWriter

class ChunkedCorpusWriter(CorpusWriter):
	def chunked_sent_string(self, sent):
		parts = []
		
		for word, tag in sent:
			try:
				brack = word in u'[]'
			except:
				brack = False
			
			if brack:
				# brackets don't get a tag
				parts.append(word)
			else:
				# make sure no brackets or slashes in tag
				tag = tag.replace(u'[', u'(').replace(u']', u')').replace(u'/', '|')
				parts.append(tuple2str((word, tag)))
		
		return ' '.join(parts)
	
	def write_sents(self, sents, *args, **kwargs):
		first = True
		
		for sent in sents:
			if not first:
				self.write(' ', *args, **kwargs)
			else:
				first = False
			
			self.write(self.chunked_sent_string(sent), *args, **kwargs)
	
	def write_paras(self, paras, *args, **kwargs):
		first = True
		
		for para in paras:
			if not first:
				self.write('\n\n', *args, **kwargs)
			else:
				first = False
			
			self.write_sents(para, *args, **kwargs)