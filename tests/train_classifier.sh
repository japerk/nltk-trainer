#!/usr/bin/env roundup

describe "train_classifier.py"

it_displays_usage_when_no_arguments() {
	./train_classifier.py 2>&1 | grep -q "usage: train_classifier.py"
}

it_cannot_find_foo() {
	last_line=$(./train_classifier.py foo 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: cannot find corpus path for foo"
}

it_cannot_import_reader() {
	last_line=$(./train_classifier.py corpora/movie_reviews --reader nltk.corpus.reader.Foo 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'module' object has no attribute 'Foo'"
}

it_trains_movie_reviews_paras() {
	test "$(./train_classifier.py movie_reviews --no-pickle --no-eval --fraction 0.5 --instances paras)" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
1000 training feats, 1000 testing feats
training NaiveBayes classifier"
}

it_trains_corpora_movie_reviews_paras() {
	test "$(./train_classifier.py corpora/movie_reviews --no-pickle --no-eval --fraction 0.5 --instances paras)" "=" "loading corpora/movie_reviews
2 labels: ['neg', 'pos']
1000 training feats, 1000 testing feats
training NaiveBayes classifier"	
}

it_cross_fold_validates() {
	folds=$(./train_classifier.py movie_reviews --cross-fold 3 2>&1|grep "training NaiveBayes classifier" -c)
	test $folds -eq 3
}

it_trains_movie_reviews_sents() {
	test "$(./train_classifier.py movie_reviews --no-pickle --no-eval --fraction 0.5 --instances sents)" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
33880 training feats, 33878 testing feats
training NaiveBayes classifier"
}

it_trains_movie_reviews_maxent() {
	last_line=$(./train_classifier.py movie_reviews --classifier Maxent --no-pickle --no-eval --fraction 0.5 --instances paras 2>&1 | tail -n 1)
	test "$last_line" "=" "training Maxent classifier"
}

it_shows_most_informative() {
	first_lines=$(./train_classifier.py movie_reviews --show-most-informative 5 --no-pickle --no-eval --fraction 0.5 | head -n 6)
	test "$first_lines" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
1000 training feats, 1000 testing feats
training NaiveBayes classifier
5 most informative features
Most Informative Features"
}