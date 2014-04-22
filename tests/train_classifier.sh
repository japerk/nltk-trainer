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
using bag of words feature extraction
1000 training feats, 1000 testing feats
training NaiveBayes classifier"
}

it_trains_corpora_movie_reviews_paras() {
	test "$(./train_classifier.py corpora/movie_reviews --no-pickle --no-eval --fraction 0.5 --instances paras)" "=" "loading corpora/movie_reviews
2 labels: ['neg', 'pos']
using bag of words feature extraction
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
using bag of words feature extraction
33880 training feats, 33878 testing feats
training NaiveBayes classifier"
}

it_shows_most_informative() {
	first_lines=$(./train_classifier.py movie_reviews --show-most-informative 5 --no-pickle --no-eval --fraction 0.5 | head -n 7)
	test "$first_lines" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
using bag of words feature extraction
1000 training feats, 1000 testing feats
training NaiveBayes classifier
5 most informative features
Most Informative Features"
}

it_passes_parameters_to_gradient_boosting_classifier() {
	classifier_line=$(./train_classifier.py movie_reviews --classifier sklearn.GradientBoostingClassifier --no-pickle --no-eval --n_estimators 3 --learning_rate 0.9 --depth_cutoff 2 --trace 2 | grep GradientBoostingClassifier | head -n 1)
	test "$classifier_line" "=" "training sklearn.GradientBoostingClassifier with {'n_estimators': 3, 'learning_rate': 0.9, 'max_depth': 2}" 
}

it_trains_with_word_count() {
	test "$(./train_classifier.py movie_reviews --no-pickle --no-eval --fraction 0.5 --value-type int)" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
using word counts feature extraction
1000 training feats, 1000 testing feats
training NaiveBayes classifier"
}
		
it_trains_with_max_feats() {
	test "$(./train_classifier.py movie_reviews --no-pickle --no-eval --fraction 0.5 --max_feats 100)" "=" "loading movie_reviews
2 labels: ['neg', 'pos']
calculating word scores
using bag of words from known set feature extraction
100 words meet min_score and/or max_feats
1000 training feats, 1000 testing feats
training NaiveBayes classifier"
}

it_trains_multi_binary() {
	test "$(./train_classifier.py problem_reports --cat_pattern '([a-z]*)' --instances sents --multi --binary --no-pickle | sed 's/[01]\.[0-9][0-9]*/<pct>/g')" "=" "loading problem_reports
5 labels: ['apache', 'eclipse', 'firefox', 'linux', 'openoffice']
using bag of words feature extraction
371 training feats, 371 testing feats
training multi-binary ['NaiveBayes'] classifier
training NaiveBayes classifier
training NaiveBayes classifier
training NaiveBayes classifier
training NaiveBayes classifier
training NaiveBayes classifier
accuracy: <pct>
average masi distance: <pct>
apache precision: <pct>
apache recall: <pct>
apache f-measure: <pct>
eclipse precision: <pct>
eclipse recall: <pct>
eclipse f-measure: <pct>
firefox precision: <pct>
firefox recall: <pct>
firefox f-measure: <pct>
linux precision: <pct>
linux recall: <pct>
linux f-measure: <pct>
openoffice precision: <pct>
openoffice recall: <pct>
openoffice f-measure: <pct>"
}

