import argparse
from nltk.misc import babelfish

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Translate a corpus')

parser.add_argument('source_corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('target_corpus', help='corpus name/path relative to an nltk_data directory')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')

# TODO: args for source & target language, corpus to translate, name of translated corpus
# open every corpus file and translate each paragraph (or just raw text) then write back out
# into new file with same name but in new corpus directory.
# NOTE babelize might be best on sentences, not paras or whole files, so do those
# one at a time.
