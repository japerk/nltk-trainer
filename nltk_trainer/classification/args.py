from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier, megam
from nltk_trainer.classification.multi import AvgProbClassifier

classifier_choices = ['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS

try:
	from .sci import ScikitsClassifier
	classifier_choices.append('Scikits')
except ImportError:
	pass

def add_maxent_args(parser):
	maxent_group = parser.add_argument_group('Maxent Classifier',
		'These options only apply when a Maxent classifier is chosen.')
	maxent_group.add_argument('--max_iter', default=10, type=int,
		help='maximum number of training iterations, defaults to %(default)d')
	maxent_group.add_argument('--min_ll', default=0, type=float,
		help='stop classification when average log-likelihood is less than this, default is %(default)d')
	maxent_group.add_argument('--min_lldelta', default=0.1, type=float,
		help='''stop classification when the change in average log-likelihood is less than this.
	default is %(default)f''')

def add_decision_tree_args(parser):
	decisiontree_group = parser.add_argument_group('Decision Tree Classifier',
		'These options only apply when the DecisionTree classifier is chosen')
	decisiontree_group.add_argument('--entropy_cutoff', default=0.05, type=float,
		help='default is 0.05')
	decisiontree_group.add_argument('--depth_cutoff', default=100, type=int,
		help='default is 100')
	decisiontree_group.add_argument('--support_cutoff', default=10, type=int,
		help='default is 10')

def make_classifier_builder(args):
	if isinstance(args.classifier, basestring):
		algos = [args.classifier]
	else:
		algos = args.classifier
	
	for algo in algos:
		if algo not in classifier_choices:
			raise ValueError('classifier %s is not supported' % algo)
	
	classifier_train_args = []
	
	for algo in algos:
		classifier_train_kwargs = {}
		
		if algo == 'DecisionTree':
			classifier_train = DecisionTreeClassifier.train
			classifier_train_kwargs['binary'] = False
			classifier_train_kwargs['entropy_cutoff'] = args.entropy_cutoff
			classifier_train_kwargs['depth_cutoff'] = args.depth_cutoff
			classifier_train_kwargs['support_cutoff'] = args.support_cutoff
			classifier_train_kwargs['verbose'] = args.trace
		elif algo == 'NaiveBayes':
			classifier_train = NaiveBayesClassifier.train
		elif algo == 'Scikits':
			classifier_train = ScikitsClassifier.train
		else:
			if algo != 'Maxent':
				classifier_train_kwargs['algorithm'] = algo
				
				if algo == 'MEGAM':
					megam.config_megam()
			
			classifier_train = MaxentClassifier.train
			classifier_train_kwargs['max_iter'] = args.max_iter
			classifier_train_kwargs['min_ll'] = args.min_ll
			classifier_train_kwargs['min_lldelta'] = args.min_lldelta
			classifier_train_kwargs['trace'] = args.trace
		
		classifier_train_args.append((algo, classifier_train, classifier_train_kwargs))
	
	def trainf(train_feats):
		classifiers = []
		
		for algo, classifier_train, train_kwargs in classifier_train_args:
			if args.trace:
				print 'training %s classifier' % algo
			
			classifiers.append(classifier_train(train_feats, **train_kwargs))
		
		if len(classifiers) == 1:
			return classifiers[0]
		else:
			return AvgProbClassifier(classifiers)
	
	return trainf
	#return lambda(train_feats): classifier_train(train_feats, **classifier_train_kwargs)