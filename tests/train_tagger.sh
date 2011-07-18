#!/usr/bin/env roundup

describe "train_tagger.py"

it_displays_usage_when_no_arguments() {
	./train_tagger.py 2>&1 | grep -q "usage: train_tagger.py"
}

it_needs_corpus_reader() {
	last_line=$(./train_tagger.py foo 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: you must specify a corpus reader"
}

it_cannot_import_reader() {
	last_line=$(./train_tagger.py foo --reader nltk.corpus.reader.Foo 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: cannot find corpus path for foo"
}

it_cannot_find_foo() {
	last_line=$(./train_tagger.py foo --reader nltk.corpus.reader.TaggedCorpusReader 2>&1 | tail -n 1)
	test "$last_line" "=" "ValueError: cannot find corpus path for foo"
}

it_trains_treebank() {
	test "$(./train_tagger.py treebank --no-pickle --no-eval --fraction 0.5)" "=" "loading treebank
3914 tagged sents, training on 1957
training AffixTagger with affix -3 and backoff <DefaultTagger: tag=-None->
training <class 'nltk.tag.sequential.UnigramTagger'> tagger with backoff <AffixTagger: size=2026>
training <class 'nltk.tag.sequential.BigramTagger'> tagger with backoff <UnigramTagger: size=3251>
training <class 'nltk.tag.sequential.TrigramTagger'> tagger with backoff <BigramTagger: size=1265>"
}

it_trains_corpora_treebank_tagged() {
	test "$(./train_tagger.py corpora/treebank/tagged --reader nltk.corpus.reader.ChunkedCorpusReader --no-pickle --no-eval --fraction 0.5)" "=" "loading corpora/treebank/tagged
51002 tagged sents, training on 25501
training AffixTagger with affix -3 and backoff <DefaultTagger: tag=-None->
training <class 'nltk.tag.sequential.UnigramTagger'> tagger with backoff <AffixTagger: size=1810>
training <class 'nltk.tag.sequential.BigramTagger'> tagger with backoff <UnigramTagger: size=3212>
training <class 'nltk.tag.sequential.TrigramTagger'> tagger with backoff <BigramTagger: size=1161>"
}

it_trains_ub() {
	test "$(./train_tagger.py treebank --sequential ub --no-pickle --no-eval --fraction 0.5)" "=" "loading treebank
3914 tagged sents, training on 1957
training <class 'nltk.tag.sequential.UnigramTagger'> tagger with backoff <DefaultTagger: tag=-None->
training <class 'nltk.tag.sequential.BigramTagger'> tagger with backoff <UnigramTagger: size=8435>"
}