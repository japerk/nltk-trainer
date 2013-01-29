#!/usr/bin/env python
import argparse, csv, os.path
import nltk_trainer.classification.corpus
from nltk_trainer import load_corpus_reader

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Dump a classified corpus to CSV')

parser.add_argument('corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('--filename', default='', help='''filename/path for where to
	store the CSV. The default is the "basename_instances.csv" where basename is
	the corpus name or the basename of the corpus path, and instances is one of
	sents, paras, or file, as given by the --instances argument.''')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')

corpus_group = parser.add_argument_group('Classified Corpus')
corpus_group.add_argument('--instances', default='paras',
	choices=('sents', 'paras', 'files'),
	help='''the group of words that represents a single training instance,
	the default is to use entire files''')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''The fraction of the corpus to use for training a binary or
	multi-class classifier, the rest will be used for evaulation.
	The default is to use the entire corpus, and to test the classifier
	against the same training data. Any number < 1 will test against
	the remaining fraction.''')

args = parser.parse_args()

###################
## corpus reader ##
###################

if args.trace:
	print 'loading corpus %s' % args.corpus

corpus = load_corpus_reader(args.corpus)

methods = {
	'sents': nltk_trainer.classification.corpus.category_sent_strings,
	'paras': nltk_trainer.classification.corpus.category_para_strings,
	'files': nltk_trainer.classification.corpus.category_file_strings
}

cat_instances = methods[args.instances](corpus)

################
## CSV output ##
################

filename = args.filename

if not filename:
	filename = '%s_%s.csv' % (os.path.basename(args.corpus), args.instances)

if args.trace:
	print 'writing to %s' % filename

with open(filename, 'w') as f:
	w = csv.writer(f, quoting=csv.QUOTE_ALL)
	
	for cat, text in cat_instances:
		w.writerow([cat, text])
