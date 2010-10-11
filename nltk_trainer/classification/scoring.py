import collections
from nltk.metrics import masi_distance
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