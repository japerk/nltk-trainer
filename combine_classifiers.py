#!/usr/bin/python
import argparse, os.path
import nltk.data
from nltk_trainer import dump_object
from nltk_trainer.classification import multi

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Combine NLTK Classifiers')
parser.add_argument('classifiers', nargs='+',
	help='one or more pickled classifiers to load and combine')
parser.add_argument('filename', default='~/nltk_data/classifiers/combined.pickle',
	help='Filename to pickle combined classifier, defaults to %(default)s')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')

args = parser.parse_args()

#########################
## combine classifiers ##
#########################

# TODO: support MaxVote combining, Hierarchical combinations

classifiers = []

for name in args.classifiers:
	if args.trace:
		print 'loading %s' % name
	
	classifiers.append(nltk.data.load(name))

combined = multi.AvgProbClassifier(classifiers)

##############################
## dump combined classifier ##
##############################

fname = os.path.expanduser(args.filename)
dump_object(combined, fname, trace=args.trace)