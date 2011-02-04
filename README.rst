NLTK Trainer
------------

NLTK Trainer exists to make training and evaluating NLTK objects as easy as possible.


Requirements
------------

You must have Python 2.6 with `argparse <http://docs.python.org/library/argparse.html>`_ and `NLTK <http://www.nltk.org/>`_ 2.0 installed. `NumPy <http://numpy.scipy.org/>`_, `SciPy <http://www.scipy.org/>`_, and `megam <http://www.cs.utah.edu/~hal/megam/>`_ are recommended for training Maxent classifiers.


Training Classifiers
--------------------

Example usage with the movie_reviews corpus can be found in `Training Binary Text Classifiers with NLTK Trainer <http://streamhacker.com/2010/10/25/training-binary-text-classifiers-nltk-trainer/>`_.

For a complete list of usage options::
	python train_classifier.py --help

Train a binary NaiveBayes classifier on the movie_reviews corpus, using paragraphs as the training instances::
	python train_classifier.py --instances paras --algorithm NaiveBayes movie_reviews

Include bigrams as features::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams movie_reviews

Minimum score threshold::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams --min_score 3 movie_reviews

Maximum number of features::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams --max_feats 1000 movie_reviews

Use the default Maxent algorithm::
	python train_classifier.py --instances paras --algorithm Maxent movie_reviews

Use the MEGAM Maxent algorithm::
	python train_classifier.py --instances paras --algorithm MEGAM movie_reviews

Train on files instead of paragraphs::
	python train_classifier.py --instances files --algorithm MEGAM movie_reviews

Train on sentences::
	python train_classifier.py --instances sents --algorithm MEGAM movie_reviews

Evaluate the classifier by training on 3/4 of the paragraphs and testing against the remaing 1/4, without pickling::
	python train_classifier.py --instances paras --algorithm NaiveBayes --fraction 0.75 --no-pickle movie_reviews


Training Part of Speech Taggers
-------------------------------

The ``train_tagger.py`` script can use any corpus included with NLTK that implements a ``tagged_sents()`` method. It can also train on the ``timit`` corpus, which includes tagged sentences that are not available through the ``TimitCorpusReader``.

For a complete list of usage options::
	python train_tagger.py --help

Train the default sequential backoff tagger on the treebank corpus::
	python train_tagger.py treebank

To use a brill tagger with the default initial tagger::
	python train_tagger.py treebank --brill

To train a NaiveBayes classifier based tagger, without a sequential backoff tagger::
	python train_tagger.py treebank --sequential '' --classifier NaiveBayes

To train a unigram tagger::
	python train_tagger.py treebank --sequential u

To train on the switchboard corpus::
	python train_tagger.py switchboard


Analyzing Tagger Coverage
-------------------------

The ``analyze_tagger_coverage.py`` script will run a part-of-speech tagger on a corpus to determine how many times each tag is found. Here's an example using the NLTK default tagger on the treebank corpus::
	python analyze_tagger_coverage.py treebank

To get detailed metrics on each tag, you can use the --metrics option. This requires using a tagged corpus in order to compare actual tags against tags found by the tagger::
	python analyze_tagger_coverage.py treebank --metrics

To use analyze the coverage of a different tagger, use the --tagger option with a path to the pickled tagger::
	python analyze_tagger_coverage.py treebank --tagger /path/to/tagger.pickle

For a complete list of usage options::
	python analyze_tagger_coverage.py --help


Analyzing a Tagged Corpus
-------------------------

The ``analyze_tagged_corpus.py`` script will show the following statistics about a tagged corpus:
* total number of words
* number of unique words
* number of tags
* the number of times each tag occurs

To analyze the treebank corpus::
	python analyze_tagged_corpus.py treebank

To sort the output by tag count from highest to lowest::
	python analyze_tagged_corpus.py treebank --sort count --reverse

To see simplified tags, instead of standard tags::
	python analyze_tagged_corpus.py treebank --simplify_tags