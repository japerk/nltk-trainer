import collections, random
from numpy import array
from nltk.metrics import masi_distance, f_measure, precision, recall
from nltk.probability import FreqDist, ConditionalFreqDist

def sum_category_word_scores(categorized_words, score_fn):
	word_fd = FreqDist()
	category_word_fd = ConditionalFreqDist()
	
	for category, words in categorized_words:
		for word in words:
			word_fd.inc(word)
			category_word_fd[category].inc(word)
	
	scores = collections.defaultdict(int)
	n_xx = category_word_fd.N()
	
	for category in category_word_fd.conditions():
		n_xi = category_word_fd[category].N()
		
		for word, n_ii in category_word_fd[category].iteritems():
			n_ix = word_fd[word]
			scores[word] += score_fn(n_ii, (n_ix, n_xi), n_xx)
	
	return scores

def sorted_word_scores(wsdict):
	return sorted(wsdict.items(), key=lambda (w, s): s, reverse=True)

def ref_test_sets(classifier, test_feats):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	
	for i, (feat, label) in enumerate(test_feats):
		refsets[label].add(i)
		observed = classifier.classify(feat)
		testsets[observed].add(i)
	
	return refsets, testsets

def multi_ref_test_sets(multi_classifier, multi_label_feats):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	
	for i, (feat, labels) in enumerate(multi_label_feats):
		for label in labels:
			refsets[label].add(i)
		
		for label in multi_classifier.classify(feat):
			testsets[label].add(i)
	
	return refsets, testsets

def avg_masi_distance(multi_classifier, multi_label_feats):
	mds = []
	
	for feat, labels in multi_label_feats:
		mds.append(masi_distance(labels, multi_classifier.classify(feat)))
	
	if mds:
		return float(sum(mds)) / len(mds)
	else:
		return 0.0

def cross_fold(instances, trainf, testf, folds=10, trace=1, metrics=True, informative=0):
	if folds < 2:
		raise ValueError('must have at least 3 folds')
	# ensure isn't an exhaustible iterable
	instances = list(instances)
	# randomize so get an even distribution, in case labeled instances are
	# ordered by label
	random.shuffle(instances)
	l = len(instances)
	step = l / folds
	
	if trace:
		print 'step %d over %d folds of %d instances' % (step, folds, l)
	
	accuracies = []
	precisions = collections.defaultdict(list)
	recalls = collections.defaultdict(list)
	f_measures = collections.defaultdict(list)
	
	for f in range(folds):
		if trace:
			print '\nfold %d' % (f+1)
			print '-----%s' % ('-'*len('%s' % (f+1)))
		
		start = f * step
		end = start + step
		train_instances = instances[:start] + instances[end:]
		test_instances = instances[start:end]
		
		if trace:
			print 'training on %d:%d + %d:%d' % (0, start, end, l)
		
		obj = trainf(train_instances)
		
		if trace:
			print 'testing on %d:%d' % (start, end)
		
		if metrics:
			refsets, testsets = ref_test_sets(obj, test_instances)
			
			for key in set(refsets.keys() + testsets.keys()):
				ref = refsets[key]
				test = testsets[key]
				p = precision(ref, test) or 0
				r = recall(ref, test) or 0
				f = f_measure(ref, test) or 0
				precisions[key].append(p)
				recalls[key].append(r)
				f_measures[key].append(f)
				
				if trace:
					print '%s precision: %f' % (key, p)
					print '%s recall: %f' % (key, r)
					print '%s f-measure: %f' % (key, f)
		
		accuracy = testf(obj, test_instances)
		
		if trace:
			print 'accuracy: %f' % accuracy
		
		accuracies.append(accuracy)
		
		if trace and informative and hasattr(obj, 'show_most_informative_features'):
			obj.show_most_informative_features(informative)
	
	if trace:
		print '\nmean and variance across folds'
		print '------------------------------'
		print 'accuracy mean: %f' % (sum(accuracies) / folds)
		print 'accuracy variance: %f' % array(accuracies).var()
		
		for key, ps in precisions.iteritems():
			print '%s precision mean: %f' % (key, sum(ps) / folds)
			print '%s precision variance: %f' % (key, array(ps).var())
		
		for key, rs in recalls.iteritems():
			print '%s recall mean: %f' % (key, sum(rs) / folds)
			print '%s recall variance: %f' % (key, array(rs).var())
		
		for key, fs in f_measures.iteritems():
			print '%s f_measure mean: %f' % (key, sum(fs) / folds)
			print '%s f_measure variance: %f' % (key, array(fs).var())
	
	return accuracies, precisions, recalls, f_measures