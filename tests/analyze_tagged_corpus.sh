#!/usr/bin/env roundup

describe "analyze_tagged_corpus.py"

it_displays_usage_when_no_arguments() {
	./analyze_tagged_corpus.py 2>&1 | grep -q "usage: analyze_tagged_corpus.py"
}

it_needs_a_tagged_corpus() {
	last_line=$(./analyze_tagged_corpus.py movie_reviews 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'CategorizedPlaintextCorpusReader' object has no attribute 'tagged_words'"
}

it_analyzes_treebank() {
	first_lines=$(./analyze_tagged_corpus.py treebank 2>&1 | head -n 4)
	test "$first_lines" "=" "loading treebank
100676 total words
12408 unique words
46 tags"
}

it_needs_corpus_reader() {
	last_line=$(./analyze_tagged_corpus.py corpora/treebank/tagged 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: you must specify a corpus reader"
}

it_needs_tagged_words() {
	last_line=$(./analyze_tagged_corpus.py corpora/treebank/tagged --reader nltk.corpus.reader.PlaintextCorpusReader 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'PlaintextCorpusReader' object has no attribute 'tagged_words'"
}

it_anayzes_treebank_tagged() {
	first_lines=$(./analyze_tagged_corpus.py corpora/treebank/tagged --reader nltk.corpus.reader.ChunkedCorpusReader 2>&1 | head -n 5)
	test "$first_lines" "=" "loading corpora/treebank/tagged
95172 total words
11994 unique words
47 tags"
}

it_anayzes_treebank_simplified_tags() {
	first_lines=$(./analyze_tagged_corpus.py treebank --simplify_tags 2>&1 | head -n 5)
	test "$first_lines" "=" "loading treebank
100676 total words
12408 unique words
31 tags"
}

it_analyzes_treebank_sort_count_reverse() {
	two_lines=$(./analyze_tagged_corpus.py treebank --sort count --reverse 2>&1 | head -n 9 | tail -n 2)
	test "$two_lines" "=" "NN           13166
IN            9857"
}