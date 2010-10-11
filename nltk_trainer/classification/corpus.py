import itertools

def categorized_words(categorized_corpus):
	for category in categorized_corpus.categories():
		yield category, categorized_corpus.words(categories=[category])

def category_fileidset(categorized_corpus, category):
	return set(categorized_corpus.fileids(categories=[category]))

def not_category_fileidset(categorized_corpus, category):
	all_fileids = set(categorized_corpus.fileids())
	good_fileids = category_fileidset(categorized_corpus, category)
	return all_fileids - good_fileids

def categorized_sent_words(categorized_corpus, category):
	return categorized_corpus.sents(categories=[category])

def not_category_sent_words(categorized_corpus, category):
	for fileid in not_category_fileidset(categorized_corpus, category):
		sents = categorized_corpus.sents(fileids=[fileid], categories=[category])
		yield itertools.chain(*sents)

def categorized_para_words(categorized_corpus, category):
	for para in categorized_corpus.paras(categories=[category]):
		yield itertools.chain(*para)

def not_category_para_words(categorized_corpus, category):
	for fileid in not_category_fileidset(categorized_corpus, category):
		for para in categorized_corpus.paras(fileids=[fileid], categories=[category]):
			yield itertools.chain(*para)

def categorized_file_words(categorized_corpus, category):
	for fileid in category_fileidset(categorized_corpus, category):
		yield itertools.chain(*categorized_corpus.sents(fileids=[fileid]))

def not_category_file_words(categorized_corpus, category):
	for fileid in not_category_fileidset(categorized_corpus, category):
		yield itertools.chain(*categorized_corpus.sents(fileids=[fileid]))