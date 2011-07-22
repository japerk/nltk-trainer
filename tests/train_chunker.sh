#!/usr/bin/env roundup

describe "train_chunker.py"

it_displays_usage_when_no_arguments() {
	./train_chunker.py 2>&1 | grep -q "usage: train_chunker.py"
}

it_needs_corpus_reader() {
	last_line=$(./train_chunker.py foo 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: you must specify a corpus reader"
}

it_cannot_import_reader() {
	last_line=$(./train_chunker.py corpora/treebank/tagged --reader nltk.corpus.reader.Foo 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'module' object has no attribute 'Foo'"
}

it_cannot_find_foo() {
	last_line=$(./train_chunker.py foo --reader nltk.corpus.reader.ChunkedCorpusReader 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: cannot find corpus path for foo"
}

it_trains_treebank_chunk() {
	test "$(./train_chunker.py treebank_chunk --no-pickle --no-eval --fraction 0.5)" "=" "loading treebank_chunk
4009 chunks, training on 2005
training ub TagChunker"
}

it_trains_treebank_chunk_u() {
	test "$(./train_chunker.py treebank_chunk --sequential u --no-pickle --no-eval --fraction 0.5)" "=" "loading treebank_chunk
4009 chunks, training on 2005
training u TagChunker"
}

it_trains_corpora_treebank_tagged() {
	test "$(./train_chunker.py corpora/treebank/tagged --reader nltk.corpus.reader.ChunkedCorpusReader --no-pickle --no-eval --fraction 0.5)" "=" "loading corpora/treebank/tagged
51002 chunks, training on 25501
training ub TagChunker"
}

it_trains_treebank_chunk_naive_bayes_classifier() {
	test "$(./train_chunker.py treebank_chunk --classifier NaiveBayes --no-pickle --no-eval --fraction 0.5)" "=" "loading treebank_chunk
4009 chunks, training on 2005
training ClassifierChunker with ['NaiveBayes'] classifier
Constructing training corpus for classifier.
Training classifier (48403 instances)
training NaiveBayes classifier"
}

it_trains_conll2000() {
	test "$(./train_chunker.py conll2000 --no-pickle --no-eval --fraction 0.5)" "=" "loading conll2000
10948 chunks, training on 5474
training ub TagChunker"
}