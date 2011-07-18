#!/usr/bin/env roundup

describe "train_classifier.py"

it_displays_usage_when_no_arguments() {
	./train_classifier.py 2>&1 | grep -q "usage: train_classifier.py"
}

it_trains_movie_reviews_paras() {
	test "$(./train_classifier.py movie_reviews --instances paras --no-pickle --no-eval --fraction 0.5)" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
1000 training feats, 1000 testing feats
training NaiveBayes classifier"
}

it_trains_corpora_movie_reviews_paras() {
	test "$(./train_classifier.py corpora/movie_reviews --instances paras --no-pickle --no-eval --fraction 0.5)" "=" "loading corpora/movie_reviews
2 labels: ['neg', 'pos']
1000 training feats, 1000 testing feats
training NaiveBayes classifier"	
}

it_cannot_find_foo() {
	last_line=$(./train_classifier.py foo 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: cannot find corpus path for foo"
}