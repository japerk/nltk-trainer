#!/usr/bin/env python
import argparse, os, os.path
import nltk.data
from nltk.misc import babelfish
from nltk_trainer import import_attr, load_corpus_reader, join_words, translate

langs = [l.lower() for l in babelfish.available_languages]

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Translate a corpus')

parser.add_argument('source_corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('target_corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('-s', '--source', default='english', choices=langs, help='source language')
parser.add_argument('-t', '--target', choices=langs, help='target language')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')
parser.add_argument('--retries', default=3, type=int,
	help='Number of babelfish retries before quiting')
parser.add_argument('--sleep', default=3, type=int,
	help='Sleep time between retries')

# TODO: these are all shared with train_classifier.py and probably others, so abstract
corpus_group = parser.add_argument_group('Input Corpus')
corpus_group.add_argument('--reader',
	default='nltk.corpus.reader.PlaintextCorpusReader',
	help='Full module path to a corpus reader class, such as %(default)s')
corpus_group.add_argument('--word-tokenizer', default='', help='Word Tokenizer class path')
corpus_group.add_argument('--sent-tokenizer', default='', help='Sent Tokenizer data.pickle path')
corpus_group.add_argument('--para-block-reader', default='', help='Block reader function path')

args = parser.parse_args()

###################
## corpus reader ##
###################

reader_args = []
reader_kwargs = {}

if args.word_tokenizer:
	reader_kwargs['word_tokenizer'] = import_attr(args.word_tokenizer)()

if args.sent_tokenizer:
	reader_kwargs['sent_tokenizer'] = nltk.data.LazyLoader(args.sent_tokenizer)

if args.para_block_reader:
	reader_kwargs['para_block_reader'] = import_attr(args.para_block_reader)

if args.trace:
	print 'loading %s' % args.source_corpus

input_corpus = load_corpus_reader(args.source_corpus, args.reader,
	*reader_args, **reader_kwargs)

#################
## translation ##
#################

for fileid in input_corpus.fileids():
	# TODO: use ~/nltk_data/corpora as dir prefix?
	path = os.path.join(args.target_corpus, fileid)
	dirname = os.path.dirname(path)
	
	if not os.path.exists(dirname):
		if args.trace:
			print 'making directory %s' % dirname
		
		os.makedirs(dirname)
	
	with open(path, 'w') as outf:
		if args.trace:
			print 'translating file %s to %s' % (fileid, path)
		
		for para in input_corpus.paras(fileids=[fileid]):
			for sent in para:
				# TODO: use intelligent joining (with punctuation)
				text = join_words(sent)
				if not text: continue
				trans = translate(text, args.source, args.corpus, trace=args.trace,
					sleep=args.sleep, retries=args.retries)
				if not trans: continue
				
				if args.trace > 1:
					print text, '-->>', trans
				
				outf.write(trans + ' ')
			
			outf.write('\n\n')