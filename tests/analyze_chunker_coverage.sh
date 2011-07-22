#!/usr/bin/env roundup

describe "analyze_chunker_coverage.py"

it_displays_usage_when_no_arguments() {
	./analyze_chunker_coverage.py 2>&1 | grep -q "usage: analyze_chunker_coverage.py"
}

it_analyzes_treebank() {
	first_lines=$(./analyze_chunker_coverage.py treebank --fraction 0.5 2>&1 | head -n 5)
	test "$first_lines" "=" "loading tagger taggers/maxent_treebank_pos_tagger/english.pickle
loading chunker chunkers/maxent_ne_chunker/english_ace_multiclass.pickle
analyzing chunker coverage of treebank with NEChunkParser

    IOB         Found  "
}

it_needs_a_corpus_reader() {
	last_line=$(./analyze_chunker_coverage.py corpora/treebank/tagged 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: you must specify a corpus reader"
}

it_cannot_import_corpus_reader() {
	last_line=$(./analyze_chunker_coverage.py corpora/treebank/tagged --reader nltk.corpus.Foo 2>&1 | tail -n 1)
	test "$last_line" "=" "AttributeError: 'module' object has no attribute 'Foo'"
}

it_analyzes_treebank_tagged() {
	first_lines=$(./analyze_chunker_coverage.py corpora/treebank/tagged --reader nltk.corpus.reader.ChunkedCorpusReader --fraction 0.5 2>&1 | head -n 3)
	test "$first_lines" "=" "loading tagger taggers/maxent_treebank_pos_tagger/english.pickle
loading chunker chunkers/maxent_ne_chunker/english_ace_multiclass.pickle
analyzing chunker coverage of corpora/treebank/tagged with NEChunkParser"
}

it_does_not_support_scoring() {
	last_line=$(./analyze_chunker_coverage.py treebank --score 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: treebank does not support scoring"
}

it_scores_treebank_chunk() {
	first_lines=$(./analyze_chunker_coverage.py treebank_chunk --score --fraction 0.5 2>&1 | head -n 5)
	test "$first_lines" "=" "loading tagger taggers/maxent_treebank_pos_tagger/english.pickle
loading chunker chunkers/maxent_ne_chunker/english_ace_multiclass.pickle
evaluating chunker score

ChunkParse score:"
}