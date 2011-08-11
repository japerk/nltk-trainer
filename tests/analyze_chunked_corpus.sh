#!/usr/bin/env roundup

describe "analyze_chunked_corpus.py"

it_displays_usage_when_no_arguments() {
	./analyze_chunked_corpus.py 2>&1 | grep -q "usage: analyze_chunked_corpus.py"
}

it_needs_a_chunked_corpus() {
	last_line=$(./analyze_chunked_corpus.py brown 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'CategorizedTaggedCorpusReader' object has no attribute 'chunked_words'"
}

it_anayzes_treebank_chunk() {
	first_lines=$(./analyze_chunked_corpus.py treebank_chunk 2>&1 | head -n 5)
	test "$first_lines" "=" "loading treebank_chunk
94200 total words
11993 unique words
46 tags
1 IOBs"
}

it_needs_corpus_reader() {
	last_line=$(./analyze_chunked_corpus.py corpora/treebank/tagged 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: you must specify a corpus reader"
}

it_needs_chunked_words() {
	last_line=$(./analyze_chunked_corpus.py corpora/treebank/tagged --reader nltk.corpus.reader.PlaintextCorpusReader 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'PlaintextCorpusReader' object has no attribute 'chunked_words'"
}

it_anayzes_treebank_tagged() {
	first_lines=$(./analyze_chunked_corpus.py corpora/treebank/tagged --reader nltk.corpus.reader.ChunkedCorpusReader 2>&1 | head -n 5)
	test "$first_lines" "=" "loading corpora/treebank/tagged
95172 total words
11994 unique words
47 tags
1 IOBs"
}

it_analyzes_treebank_chunk_sort_count_reverse() {
	two_lines=$(./analyze_chunked_corpus.py treebank_chunk --sort count --reverse 2>&1 | head -n 10 | tail -n 2)
	test "$two_lines" "=" "NN           13181   12832
IN            9970      26"
}