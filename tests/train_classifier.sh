#!/usr/bin/env roundup

describe "train_classifier.py"

before() {
	para_out="loading movie_reviews
2 labels: ['neg', 'pos']
1000 training feats, 1000 testing feats
training NaiveBayes classifier"
}

it_displays_usage_when_no_arguments() {
	./train_classifier.py 2>&1 | grep -q "usage: train_classifier.py"
}

it_trains_movie_reviews_paras() {
	test "$(./train_classifier.py movie_reviews --instances paras --no-pickle --no-eval --fraction 0.5)" "=" "$para_out"
}