#!/usr/bin/env python
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
parser.add_argument('--hierarchy', nargs='+', default=[],
	help='''Mapping of labels to classifier pickle paths to specify a classification hierarchy, such as
	"-h neutral:classifiers/movie_reviews.pickle"
	''')

args = parser.parse_args()

#####################
## AvgProb combine ##
#####################

# TODO: support MaxVote combining

classifiers = []

for name in args.classifiers:
	if args.trace:
		print 'loading %s' % name
	
	classifiers.append(nltk.data.load(name))

combined = multi.AvgProbClassifier(classifiers)

##########################
## Hierarchical combine ##
##########################

labels = combined.labels()
label_classifiers = {}

for h in args.hierarchy:
	label, path = h.split(':')
	
	if label not in labels:
		raise ValueError('%s is not in root labels: %s' % (label, labels))
	
	label_classifiers[label] = nltk.data.load(path)
	
	if args.trace:
		print 'mapping %s to %s from %s' % (label, label_classifiers[label], path)

if label_classifiers:
	if args.trace:
		'combining %d label classifiers for root %s' % (len(label_classifiers), combined)
	
	combined = multi.HierarchicalClassifier(combined, label_classifiers)

##############################
## dump combined classifier ##
##############################

fname = os.path.expanduser(args.filename)
dump_object(combined, fname, trace=args.trace)