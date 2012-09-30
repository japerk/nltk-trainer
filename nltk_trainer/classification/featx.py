import math
from nltk import probability

def bag_of_words(words):
	return dict([(word, True) for word in words])

def bag_of_words_in_set(words, wordset):
	return bag_of_words(set(words) & wordset)

def word_counts(words):
	return dict(probability.FreqDist((w, 1) for w in words))

def word_counts_in_set(words, wordset):
	return word_counts((w for w in words if w in wordset))

def train_test_feats(label, instances, featx=bag_of_words, fraction=0.75):
	labeled_instances = [(featx(i), label) for i in instances]
	
	if fraction != 1.0:
		l = len(instances)
		cutoff = int(math.ceil(l * fraction))
		return labeled_instances[:cutoff], labeled_instances[cutoff:]
	else:
		return labeled_instances, labeled_instances