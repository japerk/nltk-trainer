NLTK Trainer
------------

NLTK Trainer exists to make training and evaluating NLTK objects as easy as possible.


Requirements
------------

You must have Python 2.6 with `argparse <http://pypi.python.org/pypi/argparse/>`_ and `NLTK <http://www.nltk.org/>`_ 2.0 installed. `NumPy <http://numpy.scipy.org/>`_, `SciPy <http://www.scipy.org/>`_, and `megam <http://www.cs.utah.edu/~hal/megam/>`_ are recommended for training Maxent classifiers.


Training Classifiers
--------------------

Example usage with the movie_reviews corpus can be found in `Training Binary Text Classifiers with NLTK Trainer <http://streamhacker.com/2010/10/25/training-binary-text-classifiers-nltk-trainer/>`_.

Train a binary NaiveBayes classifier on the movie_reviews corpus, using paragraphs as the training instances::
	python train_classifier.py --instances paras --classifier NaiveBayes movie_reviews

Include bigrams as features::
	python train_classifier.py --instances paras --classifier NaiveBayes --bigrams movie_reviews

Minimum score threshold::
	python train_classifier.py --instances paras --classifier NaiveBayes --bigrams --min_score 3 movie_reviews

Maximum number of features::
	python train_classifier.py --instances paras --classifier NaiveBayes --bigrams --max_feats 1000 movie_reviews

Use the default Maxent algorithm::
	python train_classifier.py --instances paras --classifier Maxent movie_reviews

Use the MEGAM Maxent algorithm::
	python train_classifier.py --instances paras --classifier MEGAM movie_reviews

Train on files instead of paragraphs::
	python train_classifier.py --instances files --classifier MEGAM movie_reviews

Train on sentences::
	python train_classifier.py --instances sents --classifier MEGAM movie_reviews

Evaluate the classifier by training on 3/4 of the paragraphs and testing against the remaing 1/4, without pickling::
	python train_classifier.py --instances paras --classifier NaiveBayes --fraction 0.75 --no-pickle movie_reviews

For a complete list of usage options::
	python train_classifier.py --help


Using a Trained Classifier
--------------------------

You can use a trained classifier by loading the pickle file using `nltk.data.load <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.data-module.html#load>`_::
	>>> import nltk.data
	>>> classifier = nltk.data.load("classifiers/NAME_OF_CLASSIFIER.pickle")

Or if your classifier pickle file is not in a ``nltk_data`` subdirectory, you can load it with `pickle.load <http://docs.python.org/library/pickle.html#pickle.load>`_::
	>>> import pickle
	>>> classifier = pickle.load(open("/path/to/NAME_OF_CLASSIFIER.pickle"))

Either method will return an object that supports the `ClassifierI interface <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.classify.api.ClassifierI-class.html>`_. 

Once you have a ``classifier`` object, you can use it to classify word features with the ``classifier.classify(feats)`` method, which returns a label::
	>>> words = ['some', 'words', 'in', 'a', 'sentence']
	>>> feats = dict([(word, True) for word in words])
	>>> classifier.classify(feats)

If you used the ``--bigrams`` option, you should include bigrams in the dictionary using `nltk.util.bigrams(words) <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.util-module.html#bigrams>`_::
	>>> from nltk.util import bigrams
	>>> words = ['some', 'words', 'in', 'a', 'sentence']
	>>> feats = dict([(word, True) for word in words + bigrams(words)])
	>>> classifier.classify(feats)

The list of words you use for creating the feature dictionary should be created by `tokenizing <http://text-processing.com/demo/tokenize/>`_ the appropriate text instances: sentences, paragraphs, or files depending on the ``--instances`` option.


Training Part of Speech Taggers
-------------------------------

The ``train_tagger.py`` script can use any corpus included with NLTK that implements a ``tagged_sents()`` method. It can also train on the ``timit`` corpus, which includes tagged sentences that are not available through the ``TimitCorpusReader``.

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

To train on a custom corpus, whose fileids end in ".pos", using a `TaggedCorpusReader <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.corpus.reader.tagged.TaggedCorpusReader-class.html>`_::
	python train_tagger.py /path/to/corpus --reader nltk.corpus.reader.tagged.TaggedCorpusReader --fileids '.+\\.pos'

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

You can also restrict the files used with the ``--fileids`` option::
	python train_tagger.py conll2000 --fileids train.txt

For a complete list of usage options::
	python train_tagger.py --help


Using a Trained Tagger
----------------------

You can use a trained tagger by loading the pickle file using `nltk.data.load <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.data-module.html#load>`_::
	>>> import nltk.data
	>>> tagger = nltk.data.load("taggers/NAME_OF_TAGGER.pickle")

Or if your tagger pickle file is not in a ``nltk_data`` subdirectory, you can load it with `pickle.load <http://docs.python.org/library/pickle.html#pickle.load>`_::
	>>> import pickle
	>>> tagger = pickle.load(open("/path/to/NAME_OF_TAGGER.pickle"))

Either method will return an object that supports the `TaggerI interface <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.tag.api.TaggerI-class.html>`_.

Once you have a ``tagger`` object, you can use it to tag sentences (or lists of words) with the ``tagger.tag(words)`` method::
	>>> tagger.tag(['some', 'words', 'in', 'a', 'sentence'])

``tagger.tag(words)`` will return a list of 2-tuples of the form ``[(word, tag)]``.


Analyzing Tagger Coverage
-------------------------

The ``analyze_tagger_coverage.py`` script will run a part-of-speech tagger on a corpus to determine how many times each tag is found.

Here's an example using the NLTK default tagger on the treebank corpus::
	python analyze_tagger_coverage.py treebank

To get detailed metrics on each tag, you can use the ``--metrics`` option. This requires using a tagged corpus in order to compare actual tags against tags found by the tagger. See `NLTK Default Tagger Treebank Tag Coverage <http://streamhacker.com/2011/01/24/nltk-default-tagger-treebank-tag-coverage/>`_ and `NLTK Default Tagger CoNLL2000 Tag Coverage <http://streamhacker.com/2011/01/25/nltk-default-tagger-conll2000-tag-coverage/>`_ for examples and statistics.

To analyze the coverage of a different tagger, use the ``--tagger`` option with a path to the pickled tagger::
	python analyze_tagger_coverage.py treebank --tagger /path/to/tagger.pickle

To analyze coverage on a custom corpus, whose fileids end in ".pos", using a `TaggedCorpusReader <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.corpus.reader.tagged.TaggedCorpusReader-class.html>`_::
	python analyze_tagger_coverage.py /path/to/corpus --reader nltk.corpus.reader.tagged.TaggedCorpusReader --fileids '.+\\.pos'

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

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

To analyze a custom corpus, whose fileids end in ".pos", using a `TaggedCorpusReader <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.corpus.reader.tagged.TaggedCorpusReader-class.html>`_::
	python analyze_tagged_corpus.py /path/to/corpus --reader nltk.corpus.reader.tagged.TaggedCorpusReader --fileids '.+\\.pos'

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

For a complete list of usage options::
	python analyze_tagged_corpus.py --help


Training IOB Chunkers
---------------------

The ``train_chunker.py`` script can use any corpus included with NLTK that implements a ``chunked_sents()`` method.

Train the default sequential backoff tagger based chunker on the treebank_chunk corpus::
	``python train_chunker.py treebank_chunk``

To train a NaiveBayes classifier based chunker::
	``python train_chunker.py treebank_chunk --classifier NaiveBayes``

To train on the conll2000 corpus::
	``python train_chunker.py conll2000``

To train on a custom corpus, whose fileids end in ".pos", using a `ChunkedCorpusReader <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.corpus.reader.chunked.ChunkedCorpusReader-class.html>`_::
	``python train_chunker.py /path/to/corpus --reader nltk.corpus.reader.chunked.ChunkedCorpusReader --fileids '.+\\.pos'``

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

You can also restrict the files used with the ``--fileids`` option::
	``python train_chunker.py conll2000 --fileids train.txt``

For a complete list of usage options::
	``python train_chunker.py --help``