import argparse

# TODO: args for input classifier, input text (as raw file), instances to use (paras || sents),
# min threshold, output corpus name
# read in each input example, prob classify, and if exceeds threshold, write to new
# corpus, one file per category

# TODO: also have optional language arg to translate input example with babelfish,
# but still write original example bsaed on classification or translated example

# TODO: arg(s) to specify categorized word list corpus instead of classifier pickle
# can have additional arguments for decision threshold. this will create a
# KeywordClassifier that can be used just like any other NLTK classifier

# TODO: if new corpus files already exist, append to them, and make sure the
# first append example is separate (enough) from the last example in the file
# (we don't want to append a paragraph right next to another paragraph, creating a single paragraph)
