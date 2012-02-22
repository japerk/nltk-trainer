
class ClassifiedCorpusWriter(CorpusWriter):
	def __init__(self, path, labels):
		self.path = path
		self.labels = labels
	# TODO: make sure works with with keyword
	def __enter__(self):
		self._files = dict([(l, self.open(os.path.join(path, l), 'a')) for l in labels])
	
	def __exit__(self):
		for f in self._files.values():
			f.close()
	
	def write(self, text, label):
		self._files[label].write(text + u'\n\n')