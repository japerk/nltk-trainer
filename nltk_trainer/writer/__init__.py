import codecs, collections, os, os.path

class CorpusWriter(object):
	def __init__(self, fileids, path='~/nltk_data/corpora', mode='a', encoding='utf-8', trace=1):
		assert fileids and path and mode
		self.mode = mode
		self.encoding = encoding
		self.trace = trace or 0
		self.full_path = os.path.expanduser(path)
		
		for dirname in set([os.path.dirname(fileid) for fileid in fileids]):
			dirpath = os.path.join(self.full_path, dirname)
			
			if not os.path.exists(dirpath):
				if trace:
					print('making directory %s' % dirpath)
				
				os.makedirs(dirpath)
		
		self.fileids = [os.path.join(self.full_path, fileid) for fileid in fileids]
		self.files = {}
	
	def get_file(self, fileid):
		if not fileid.startswith(self.full_path):
			fileid = os.path.join(self.full_path, fileid)
		
		if fileid not in self.files:
			self.files[fileid] = codecs.open(fileid, self.mode, self.encoding)
		
		return self.files[fileid]
	
	def open(self):
		for fileid in self.fileids:
			if self.trace:
				print('opening %s' % fileid)
			
			self.get_file(fileid)
		
		return self
	
	def close(self, *args, **kwargs):
		for fileid, f in self.files.items():
			if self.trace:
				print('closing %s' % fileid)
			
			f.close()
			del self.files[fileid]
	
	__enter__ = open
	__exit__ = close
	__del__ = close
	
	def write(self, s, fileid=None):
		if not fileid:
			fileid = self.fileids[0]
		
		self.get_file(fileid).write(s)