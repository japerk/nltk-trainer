from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier, megam
from nltk_trainer.classification.multi import AvgProbClassifier

classifier_choices = ['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS

dense_classifiers = set(['ExtraTreesClassifier', 'GradientBoostingClassifier',
		'RandomForestClassifier', 'GaussianNB', 'DecisionTreeClassifier'])
verbose_classifiers = set(['RandomForestClassifier', 'SVC'])

try:
	import svmlight # do this first since svm module makes ugly errors
	from nltk.classify.svm import SvmClassifier
	classifier_choices.append('Svm')
except:
	pass

try:
	from nltk.classify import scikitlearn
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.pipeline import Pipeline
	from sklearn import ensemble, feature_selection, linear_model, naive_bayes, neighbors, svm, tree
	
	classifiers = [
		ensemble.ExtraTreesClassifier,
		ensemble.GradientBoostingClassifier,
		ensemble.RandomForestClassifier,
		linear_model.LogisticRegression,
		#linear_model.SGDClassifier, # NOTE: this seems terrible, but could just be the options
		naive_bayes.BernoulliNB,
		naive_bayes.GaussianNB,
		naive_bayes.MultinomialNB,
		neighbors.KNeighborsClassifier, # TODO: options for nearest neighbors
		svm.LinearSVC,
		svm.NuSVC,
		svm.SVC,
		tree.DecisionTreeClassifier,
	]
	sklearn_classifiers = {}
	
	for classifier in classifiers:
		sklearn_classifiers[classifier.__name__] = classifier
	
	classifier_choices.extend(sorted(['sklearn.%s' % c.__name__ for c in classifiers]))
except ImportError as exc:
	sklearn_classifiers = {}

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

sklearn_kwargs = {
	# ensemble
	'ExtraTreesClassifier': ['criterion', 'max_feats', 'depth_cutoff', 'n_estimators'],
	'GradientBoostingClassifier': ['learn_rate', 'max_feats', 'depth_cutoff', 'n_estimators'],
	'RandomForestClassifier': ['criterion', 'max_feats', 'depth_cutoff', 'n_estimators'],
	# linear_model
	'LogisticRegression': ['C','penalty'],
	# naive_bayes
	'BernoulliNB': ['alpha'],
	'MultinomialNB': ['alpha'],
	# svm
	'LinearSVC': ['C', 'loss', 'penalty'],
	'NuSVC': ['nu', 'kernel'],
	'SVC': ['C', 'kernel'],
	# tree
	'DecisionTreeClassifier': ['criterion', 'max_feats', 'depth_cutoff'],
}

def add_sklearn_args(parser):
	if not sklearn_classifiers: return
	
	sklearn_group = parser.add_argument_group('sklearn Classifiers',
		'These options are used by one or more sklearn classification algorithms.')
	sklearn_group.add_argument('--alpha', type=float, default=1.0,
		help='smoothing parameter for naive bayes classifiers, default is %(default)s')
	sklearn_group.add_argument('--C', type=float, default=1.0,
		help='penalty parameter, default is %(default)s')
	sklearn_group.add_argument('--criterion', choices=['gini', 'entropy'],
		default='gini', help='Split quality function, default is %(default)s')
	sklearn_group.add_argument('--kernel', default='rbf',
		choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
		help='kernel type for support vector machine classifiers, default is %(default)s')
	sklearn_group.add_argument('--learn_rate', type=float, default=0.1,
		help='learning rate, default is %(default)s')
	sklearn_group.add_argument('--loss', choices=['l1', 'l2'],
		default='l2', help='loss function, default is %(default)s')
	sklearn_group.add_argument('--n_estimators', type=int, default=10,
		help='Number of trees for Decision Tree ensembles, default is %(default)s')
	sklearn_group.add_argument('--nu', type=float, default=0.5,
		help='upper bound on fraction of training errors & lower bound on fraction of support vectors, default is %(default)s')
	sklearn_group.add_argument('--penalty', choices=['l1', 'l2'],
		default='l2', help='norm for penalization, default is %(default)s')
	sklearn_group.add_argument('--tfidf', default=False, action='store_true',
		help='Use TfidfTransformer')

# for mapping existing args to sklearn args
sklearn_keys = {
	'max_feats': 'max_features',
	'depth_cutoff': 'max_depth'
}

def make_sklearn_classifier(algo, args):
	name = algo.split('.', 1)[1]
	kwargs = {}
	
	for key in sklearn_kwargs.get(name, []):
		val = getattr(args, key)
		if val: kwargs[sklearn_keys.get(key, key)] = val
	
	if args.trace and kwargs:
		print 'training %s with %s' % (algo, kwargs)
	
	if args.trace and name in verbose_classifiers:
		kwargs['verbose'] = True
	
	return sklearn_classifiers[name](**kwargs)

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
		elif algo == 'Svm':
			classifier_train = SvmClassifier.train
		elif algo.startswith('sklearn.'):
			# TODO: support many options for building an estimator pipeline
			pipe = [('classifier', make_sklearn_classifier(algo, args))]
			
			if args.tfidf:
				if args.trace:
					print 'using tfidf transformer with norm %s' % args.penalty
				
				pipe.insert(0, ('tfidf', TfidfTransformer(norm=args.penalty)))
			
			sparse = pipe[-1][1].__class__.__name__ not in dense_classifiers
			
			if not sparse and args.trace:
				print 'using dense matrix'
			
			if args.value_type == 'bool' and not args.tfidf:
				dtype = bool
			elif args.value_type == 'int' and not args.tfidf:
				dtype = int
			else:
				dtype = float
			
			if args.trace:
				print 'using dtype %s' % dtype.__name__
			
			classifier_train = scikitlearn.SklearnClassifier(Pipeline(pipe), dtype=dtype, sparse=sparse).train
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
