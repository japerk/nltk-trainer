import itertools

def category_words(categorized_corpus):
	for category in categorized_corpus.categories():
		yield category, categorized_corpus.words(categories=[category])

def category_fileidset(categorized_corpus, category):
	return set(categorized_corpus.fileids(categories=[category]))

def category_sent_words(categorized_corpus, category):
	return categorized_corpus.sents(categories=[category])

def category_para_words(categorized_corpus, category):
	for para in categorized_corpus.paras(categories=[category]):
		yield itertools.chain(*para)

def category_file_words(categorized_corpus, category):
	for fileid in category_fileidset(categorized_corpus, category):
		yield categorized_corpus.words(fileids=[fileid])

## multi category corpus ##

def corpus_fileid_categories(categorized_corpus, prefix):
	for fileid in categorized_corpus.fileids():
		if not prefix or fileid.startswith(prefix):
			yield fileid, set(categorized_corpus.categories(fileids=[fileid]))
	
def multi_category_sent_words(categorized_corpus, fileid_prefix=''):
	for fileid, categories in corpus_fileid_categories(categorized_corpus, fileid_prefix):
		for sent in categorized_corpus.sents(fileids=[fileid]):
			yield sent, categories

def multi_category_para_words(categorized_corpus, fileid_prefix=''):
	for fileid, categories in corpus_fileid_categories(categorized_corpus, fileid_prefix):
		for para in categorized_corpus.paras(fileids=[fileid]):
			yield itertools.chain(*para), categories

def multi_category_file_words(categorized_corpus, fileid_prefix=''):
	for fileid, categories in corpus_fileid_categories(categorized_corpus, fileid_prefix):
		yield categorized_corpus.words(fileids=[fileid]), categories

################
## csv output ##
################

def category_sent_strings(corpus):
	for cat in corpus.categories():
		for sent in corpus.sents(categories=[cat]):
			yield cat, ' '.join(sent)

def category_para_strings(corpus):
	for cat in corpus.categories():
		for para in corpus.paras(categories=[cat]):
			yield cat, ' '.join([' '.join(sent) for sent in para])

def category_file_strings(corpus):
	for cat in corpus.categories():
		for fileid in corpus.fileids(categories=[cat]):
			yield cat, corpus.raw(fileids=[fileid])