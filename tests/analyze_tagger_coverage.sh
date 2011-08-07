#!/usr/bin/env roundup

describe "analyze_tagger_coverage.py"

it_displays_usage_when_no_arguments() {
	./analyze_tagger_coverage.py 2>&1 | grep -q "usage: analyze_tagger_coverage.py"
}

it_analyzes_treebank() {
	first_lines=$(./analyze_tagger_coverage.py treebank --fraction 0.5 2>&1 | head -n 4)
	test "$first_lines" "=" "loading tagger taggers/maxent_treebank_pos_tagger/english.pickle
analyzing tag coverage of treebank with ClassifierBasedPOSTagger

  Tag      Found  "
}

it_needs_a_corpus_reader() {
	last_line=$(./analyze_tagger_coverage.py corpora/treebank/tagged 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: you must specify a corpus reader"
}

it_cannot_import_corpus_reader() {
	last_line=$(./analyze_tagger_coverage.py corpora/treebank/tagged --reader nltk.corpus.Foo 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'module' object has no attribute 'Foo'"
}

it_analyzes_treebank_tagged() {
	first_lines=$(./analyze_tagger_coverage.py corpora/treebank/tagged --reader nltk.corpus.reader.ChunkedCorpusReader --fraction 0.5 2>&1 | head -n 4)
	test "$first_lines" "=" "loading tagger taggers/maxent_treebank_pos_tagger/english.pickle
analyzing tag coverage of corpora/treebank/tagged with ClassifierBasedPOSTagger

  Tag      Found  "
}

it_does_not_support_metrics() {
	last_line=$(./analyze_tagger_coverage.py movie_reviews --metrics 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: movie_reviews does not support metrics"
}

it_analyzes_treebank_metrics() {
	two_lines=$(./analyze_tagger_coverage.py treebank --metrics --fraction 0.5 2>&1 | head -n 5 | tail -n 2)
	echo "$two_lines" | grep -q "Accuracy:"
	echo "$two_lines" | grep -q "Unknown words:"
}

it_requires_metrics_with_simplify_tags() {
	last_line=$(./analyze_tagger_coverage.py treebank --simplify_tags 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: simplify_tags can only be used with the --metrics option"
}

it_analyzes_treebank_simplify_tags_metrics() {
	two_lines=$(./analyze_tagger_coverage.py treebank --simplify_tags --metrics --fraction 0.5 2>&1 | head -n 5 | tail -n 2)
	echo "$two_lines" | grep -q "Accuracy:"
	echo "$two_lines" | grep -q "Unknown words:"
}